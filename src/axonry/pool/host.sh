#!/bin/bash
# An owned GPU host (pool/owned.py), run as root over ssh from /srv/axonry/host. Every command is idempotent.
#
#   host.sh start STEP CMD ARGS...   run `host.sh CMD ARGS...` detached once: prints ok | running | failed (+ log tail)
#   host.sh init IMAGE CHINA         Docker + the NVIDIA runtime, firewall, the jump user, sshd, the pinned image
#   host.sh bake BUNDLE IMAGE CHINA  the run image axonry-run:<bundle sha>: IMAGE + setup.sh (all a run needs, offline)
#   host.sh eval BUNDLE ITEMS IMAGE CHINA   the evaluation container (all GPUs), set up, with the rules' benchmark
#                                    items (evaluation/benchdata.py), checked
#   host.sh run N GPUS TAG ROW       a fresh run container in slot N on GPUS (e.g. 0,1,2,3), sshd on 127.0.0.1:2300+N
#   host.sh jump PORT PUBKEY         the jump user forwards this key to 127.0.0.1:PORT only (replaces the port's
#                                    previous key: a finished run's key reaches nothing)
#   host.sh state NAME               "<running> <row>" of a container, or "missing"
#   host.sh stop NAME                remove a container
#
# CHINA=1: Docker, PyPI and Ubuntu through mirrors in China (the base: ModelScope if Hugging Face is unreachable). Containers get no route to this host
# (firewall) nor to each other (icc off); the agent of a run reaches its container only through the jump user.
set -euo pipefail
R=/srv/axonry
H=$R/host
JUMP=axjump
EVAL_PORT=2222

ports() { echo "127.0.0.1:$EVAL_PORT"; for n in 1 2 3 4 5 6 7 8; do echo "127.0.0.1:$((2300 + n))"; done; }

wait_sshd() {   # the container's entrypoint (container_init.sh) installed sshd and made its host keys
  for _ in $(seq 600); do docker exec "$1" test -f /etc/ssh/.axonry-host 2>/dev/null && return 0; sleep 1; done
  echo "$1: sshd did not come up" >&2; docker logs --tail 20 "$1" >&2; return 1
}

setup_in() {    # setup.sh in a container, from the bundle (pip through the mirror in China)
  local c=$1 bundle=$2 china=$3 env=()
  [ "$china" = 1 ] && env=(-e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple)
  docker cp "$bundle" "$c:/root/axonry-setup.tar"
  docker exec "${env[@]}" "$c" bash -c 'rm -rf /root/axonry-setup && mkdir /root/axonry-setup &&
    tar xf /root/axonry-setup.tar -C /root/axonry-setup && /root/axonry-setup/setup.sh "$(nvidia-smi -L | wc -l)"'
}

install_docker() {
  local d=https://download.docker.com nv=https://nvidia.github.io/libnvidia-container
  [ "$1" = 1 ] && d=https://mirrors.tuna.tsinghua.edu.cn/docker-ce && nv=https://mirrors.ustc.edu.cn/libnvidia-container
  . /etc/os-release
  if command -v dnf >/dev/null; then
    # docker-ce packages fetched directly: dnf's own downloads from the mirrors stall on some hosts
    local url="$d/linux/centos/$(rpm -E %rhel)/x86_64/stable/Packages" tmp
    tmp=$(mktemp -d)
    local index; index=$(curl -fsS --retry 5 --retry-all-errors "$url/")
    for p in docker-ce docker-ce-cli containerd.io docker-buildx-plugin; do
      f=$(echo "$index" | grep -o "href=\"$p-[0-9][^\"]*x86_64.rpm\"" | sed 's/href="//; s/"$//' | sort -V | tail -1)
      curl -fsS --retry 5 --retry-all-errors -o "$tmp/$f" "$url/$f"
    done
    curl -fsS --retry 5 --retry-all-errors "$nv/stable/rpm/nvidia-container-toolkit.repo" |
      sed "s#https://nvidia.github.io/libnvidia-container#$nv#g" > /etc/yum.repos.d/nvidia-container-toolkit.repo
    dnf -y -q --setopt=install_weak_deps=False install "$tmp"/*.rpm nvidia-container-toolkit
    rm -rf "$tmp"
  else
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsS --retry 5 --retry-all-errors "$d/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] $d/linux/$ID $VERSION_CODENAME stable" \
      > /etc/apt/sources.list.d/docker.list
    curl -fsS --retry 5 --retry-all-errors "$nv/gpgkey" | gpg --batch --yes --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
    curl -fsS --retry 5 --retry-all-errors "$nv/stable/deb/nvidia-container-toolkit.list" | sed \
      -e "s#https://nvidia.github.io/libnvidia-container#$nv#g" \
      -e 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
      nvidia-container-toolkit
  fi
}

init() {
  local image=$1 china=$2
  # the host is ours alone: GPUs busy with anything but our containers are the operator's to free
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | while IFS=, read -r pid name; do
    grep -qE 'docker|containerd' "/proc/$pid/cgroup" 2>/dev/null || echo "$pid$name"; done)
  [ -z "$busy" ] || { echo "GPUs in use by other processes: $busy" >&2; return 1; }

  command -v docker >/dev/null && command -v nvidia-ctk >/dev/null || install_docker "$china"
  local cfg=/etc/docker/daemon.json old
  old=$(cat $cfg 2>/dev/null || true)
  mkdir -p /etc/docker
  python3 - "$cfg" "$china" <<'PY'   # our keys into the host's own settings (log limits, runtimes stay)
import json, sys
p, china = sys.argv[1], sys.argv[2] == "1"
try:
    c = json.load(open(p))
except (OSError, ValueError):
    c = {}
c["icc"] = False
if china:
    c.update({"max-concurrent-downloads": 8,
              "registry-mirrors": ["https://docker.m.daocloud.io", "https://docker.1ms.run"]})
json.dump(c, open(p, "w"), indent=2)
PY
  nvidia-ctk runtime configure --runtime=docker >/dev/null
  # no systemd reload or docker restart unless something changed: on a host with cgroup v2 and the systemd driver,
  # a reload takes the GPUs from running containers started without --device (see run)
  systemctl is-enabled -q docker || systemctl enable docker >/dev/null 2>&1
  if [ "$(cat $cfg)" != "$old" ] || ! systemctl is-active -q docker; then systemctl restart docker; fi

  # containers never open connections to this host (its sshd, its LAN services)
  local unit=/etc/systemd/system/axonry-firewall.service
  cat > $unit.new <<'EOF'
[Unit]
Description=axonry: containers open no connections to this host
After=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'iptables -C INPUT -i docker0 -m conntrack --ctstate NEW -j DROP 2>/dev/null || iptables -I INPUT 1 -i docker0 -m conntrack --ctstate NEW -j DROP'
[Install]
WantedBy=multi-user.target
EOF
  if cmp -s $unit.new $unit; then rm $unit.new; else mv $unit.new $unit && systemctl daemon-reload; fi
  systemctl is-enabled -q axonry-firewall || systemctl enable axonry-firewall >/dev/null 2>&1
  systemctl start axonry-firewall

  # the jump user: forwarding to one container port per key (authorized_keys), nothing else
  id $JUMP >/dev/null 2>&1 || useradd -m -s "$(command -v nologin)" $JUMP
  usermod -p "*" $JUMP     # key login only; a locked account ("!") is refused by sshd without PAM
  install -d -m 700 -o $JUMP -g $JUMP /home/$JUMP/.ssh
  touch /home/$JUMP/.ssh/authorized_keys
  chown $JUMP:$JUMP /home/$JUMP/.ssh/authorized_keys && chmod 600 /home/$JUMP/.ssh/authorized_keys
  local sc=/etc/ssh/sshd_config
  [ -f $sc.before-axonry ] || cp $sc $sc.before-axonry
  sed -i '/^# >>> axonry/,/^# <<< axonry/d' $sc
  cat >> $sc <<EOF
# >>> axonry (pool/host.sh): root reaches the containers, the run agents their own container, through this sshd
Match User root
    AllowTcpForwarding local
    PermitOpen $(ports | tr '\n' ' ')
Match User $JUMP
    AllowTcpForwarding local
    X11Forwarding no
    PermitTTY no
    ForceCommand /bin/false
# <<< axonry
EOF
  sshd -t
  systemctl reload sshd 2>/dev/null || systemctl reload ssh

  install -d -m 755 $R $R/models
  docker pull -q "$image" >/dev/null
}

bake() {
  local bundle=$1 image=$2 china=$3 tag
  tag=axonry-run:$(sha256sum "$bundle" | cut -c1-12)
  docker image inspect "$tag" >/dev/null 2>&1 && return 0
  docker rm -f axonry-bake >/dev/null 2>&1 || true
  docker run -d --name axonry-bake --gpus all -e AXONRY_CHINA="$china" -v $R/models:$R/models \
    -v $H/init:/srv/axonry-init:ro --entrypoint /srv/axonry-init/container_init.sh "$image" >/dev/null
  wait_sshd axonry-bake
  setup_in axonry-bake "$bundle" "$china"
  docker exec axonry-bake sh -c 'rm -rf /etc/ssh/.axonry-host /etc/axonry-deadline /root/.cache /root/axonry-setup*'
  docker commit axonry-bake "$tag" >/dev/null
  docker rm -f axonry-bake >/dev/null
  # older run images go once no container uses them (an image in use refuses removal)
  docker images axonry-run --format '{{.Repository}}:{{.Tag}}' | grep -vx "$tag" | xargs -r docker rmi >/dev/null 2>&1 || true
}

eval_container() {
  local bundle=$1 items=$2 image=$3 china=$4 sha
  sha=$(sha256sum "$bundle" | cut -c1-12)
  docker rm -f axonry-eval >/dev/null 2>&1 || true
  docker run -d --name axonry-eval --label axonry.bundle="$sha" --restart unless-stopped --gpus all --ipc=host \
    --shm-size 64g --ulimit nofile=1048576:1048576 -p 127.0.0.1:$EVAL_PORT:22 -e AXONRY_CHINA="$china" \
    -e AXONRY_ROLE=eval -v $R/models:$R/models -v $H/init:/srv/axonry-init:ro \
    --entrypoint /srv/axonry-init/container_init.sh "$image" >/dev/null
  wait_sshd axonry-eval
  setup_in axonry-eval "$bundle" "$china"
  # the benchmark items where the evaluation jobs' toolkit looks for them ($WSP_WORK/data), checked offline
  docker exec -i axonry-eval tar xf - -C $R/wsp_work < "$items"
  docker exec -e HF_HUB_OFFLINE=1 -e PYTHONPATH=$R/repos/WeightSpaceProgramming -e WSP_WORK=$R/wsp_work axonry-eval \
    python3 -m axonry.evaluation.benchdata /root/axonry-setup/benchmarks.json
}

run() {
  local n=$1 gpus=$2 tag=$3 row=$4 total ngpu cpus mem
  total=$(nvidia-smi -L | wc -l); ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
  cpus=$(( $(nproc) * ngpu / total ))
  mem=$(( $(awk '/^MemTotal/{print $2}' /proc/meminfo) * 9 / 10 * ngpu / total / 1024 ))
  # its GPUs' device nodes (by minor number) and the driver's, also as --device: they enter the container's own
  # device rules, so a systemd reload on the host (cgroup v2, systemd driver) does not take them away
  local devs=() i d
  for i in ${gpus//,/ }; do
    devs+=(--device "/dev/nvidia$(nvidia-smi -q -i "$i" | awk -F: '/Minor Number/ {gsub(/ /, "", $2); print $2}')")
  done
  for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do [ -c $d ] && devs+=(--device $d); done
  docker rm -f "axonry-run$n" >/dev/null 2>&1 || true
  docker run -d --name "axonry-run$n" --label axonry.row="$row" --gpus "\"device=$gpus\"" "${devs[@]}" --shm-size 32g \
    --cpus "$cpus" --memory "${mem}m" --ulimit nofile=1048576:1048576 -p "127.0.0.1:$((2300 + n)):22" \
    -v $R/models:$R/models -v $H/init:/srv/axonry-init:ro --entrypoint /srv/axonry-init/container_init.sh "$tag" \
    >/dev/null
  wait_sshd "axonry-run$n"
}

jump() {
  local f=/home/$JUMP/.ssh/authorized_keys
  { grep -v "permitopen=\"127.0.0.1:$1\"" $f || true
    echo "restrict,port-forwarding,permitopen=\"127.0.0.1:$1\" $2"; } > $f.new
  chown $JUMP:$JUMP $f.new && chmod 600 $f.new && mv $f.new $f
}

case "${1:-}" in
  start)
    step=$2; shift 2; cd $H
    if [ -f "$step.ok" ]; then echo ok
    elif [ -f "$step.failed" ]; then echo failed; tail -c 800 "$step.log"; rm -f "$step.failed"
    elif flock -n "$step.lock" true; then
      STEP=$step setsid nohup flock -n "$step.lock" sh -c '"$0" "$@" && touch "$STEP.ok" || touch "$STEP.failed"' "$H/host.sh" "$@" \
        > "$step.log" 2>&1 < /dev/null &
      echo running
    else echo running; fi ;;
  init) init "$2" "$3" ;;
  bake) bake "$2" "$3" "$4" ;;
  eval) eval_container "$2" "$3" "$4" "$5" ;;
  run) run "$2" "$3" "$4" "$5" ;;
  jump) jump "$2" "$3" ;;
  state) docker inspect -f '{{.State.Running}} {{index .Config.Labels "axonry.row"}}{{index .Config.Labels "axonry.bundle"}}' \
           "$2" 2>/dev/null || echo missing ;;
  stop) docker rm -f "$2" >/dev/null 2>&1 || true ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
