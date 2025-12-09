// ecosystem.config.js -- 3 miners with FSDP=2, 1 miner with single GPU, 1 validator with single GPU
require('dotenv').config({ path: '.env' });

const { execSync } = require('child_process');
const RANDOM_SUFFIX = execSync(
  "cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1"
)
  .toString()
  .trim();

const PROJECT_NAME = `test_${RANDOM_SUFFIX}`;

module.exports = {
  apps: [
    /*───────────────────────── Miner ─────────────────────────*/
    {
      name            : "TM1",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "neurons/miner.py",
        "--wallet.name", "miner0",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--subtensor.network", "ws://127.0.0.1:9944",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME,
        "--pp_degree", "2",
        "--dp_degree", "2"
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "1,2,3,4"
      }
    },
    // {
    //   name            : "TM2",
    //   exec_mode       : "fork",
    //   exec_interpreter: "none",
    //   script          : "torchrun",
    //   args: [
    //     "--standalone",
    //     "--nnodes", "1",
    //     "--nproc_per_node", "2",
    //     "neurons/miner.py",
    //     "--wallet.name", "miner1",
    //     "--wallet.hotkey", "default",
    //     "--device", "cuda",
    //     "--subtensor.network", "local",
    //     "--netuid", "2",
    //     "--use_wandb",
    //     "--project", PROJECT_NAME,
    //     "--pp_degree", "1",
    //     "--dp_degree", "2"
    //   ],
    //   env: {
    //     ...process.env,
    //     PROJECT_NAME,
    //     CUDA_VISIBLE_DEVICES: "3,4"
    //   }
    // },
    // {
    //   name            : "TM3",
    //   exec_mode       : "fork",
    //   exec_interpreter: "none",
    //   script          : "torchrun",
    //   args: [
    //     "--standalone",
    //     "--nnodes", "1",
    //     "--nproc_per_node", "1",
    //     "neurons/miner.py",
    //     "--wallet.name", "miner3",
    //     "--wallet.hotkey", "default",
    //     "--device", "cuda",
    //     "--subtensor.network", "local",
    //     "--netuid", "2",
    //     "--use_wandb",
    //     "--project", PROJECT_NAME,
    //     "--pp_degree", "1",
    //     "--dp_degree", "1"
    //   ],
    //   env: {
    //     ...process.env,
    //     PROJECT_NAME,
    //     CUDA_VISIBLE_DEVICES: "6"
    //   }
    // },
    // {
    //   name            : "TM4",
    //   exec_mode       : "fork",
    //   exec_interpreter: "none",
    //   script          : "torchrun",
    //   args: [
    //     "--standalone",
    //     "--nnodes", "1",
    //     "--nproc_per_node", "1",
    //     "neurons/miner.py",
    //     "--wallet.name", "miner4",
    //     "--wallet.hotkey", "default",
    //     "--device", "cuda",
    //     "--subtensor.network", "local",
    //     "--netuid", "2",
    //     "--use_wandb",
    //     "--project", PROJECT_NAME,
    //     "--pp_degree", "1",
    //     "--dp_degree", "1"
    //   ],
    //   env: {
    //     ...process.env,
    //     PROJECT_NAME,
    //     CUDA_VISIBLE_DEVICES: "7"
    //   }
    // },

    /*──────────────────────── Validator ──────────────────────*/
    {
      name            : "TV1",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "neurons/validator.py",
        "--wallet.name", "validator",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--subtensor.network", "ws://127.0.0.1:9944",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "0,5,6,7"
      }
    }
  ]
};
