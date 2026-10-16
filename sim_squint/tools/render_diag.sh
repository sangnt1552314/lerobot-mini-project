#!/usr/bin/env bash
# Which Vulkan setup renders on this node: ManiSkill PickCube, CPU backend, 1 camera, under three library variants.
#   bash sim_squint/tools/render_diag.sh > logs/squint/render_variants.log 2>&1
cd "$(dirname "$0")/../.."
V=/scratch/e1583535/virtualenvs/squint
echo "where=${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-host}} $(hostname)"
ldconfig -p | grep -i -E "GLX_nvidia|gpucomp|libvulkan.so"
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null
T='import gymnasium as gym, mani_skill.envs
e=gym.make("PickCube-v1",num_envs=1,obs_mode="rgb",sim_backend="cpu"); o,_=e.reset(seed=0)
print("RENDER OK", o["sensor_data"]["base_camera"]["rgb"].float().mean().item(), flush=True)'
run() { echo "=== $1"; timeout 300 bash sim_squint/env.sh bash -c "$2; python -c '$T'" 2>&1 | grep -v -i warn | tail -3; }
run "A: env.sh as is (venv libvulkan + egl_shim libGLdispatch)" "true"
run "B: SAPIEN's own libvulkan 1.3.224" "export SAPIEN_VULKAN_LIBRARY_PATH=$V/lib/python3.10/site-packages/sapien/vulkan_library/libvulkan.so.1.3.224"
run "C: no egl_shim libGLdispatch" "export LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | sed 's|/scratch/e1583535/opt/egl_shim/lib:||')"
run "D: driver's own ICD lookup (no VK_ICD_FILENAMES)" "unset VK_ICD_FILENAMES VK_DRIVER_FILES"
