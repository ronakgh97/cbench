use cbench::bencher::{MAX_RUN, MAX_WARMUP, POOL_CAPACITY};
#[allow(unused)]
use cbench::bencher::{SAMPLE_SIZE, run_benchmark};
use clap::{Parser, Subcommand};
use std::cmp::min;

#[derive(Parser)]
#[command(
    name = "cbench",
    version = "1.0.0-alpha",
    about = "BLAS, Crypto based cpu micro-benching tool"
)]
struct CliArgs {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Run {
        /// Number of runs to perform (default: 12)
        #[arg(short, long, default_value = "12")]
        runs: usize,

        /// Number of warmup runs before benchmarking (default: 2)
        #[arg(short, long, default_value = "2")]
        warmups: usize,

        /// Max threads to use in bench (default: 0, all available thread)
        #[arg(short, long, default_value = "0")]
        max_threads: usize,
    },
}
pub const MAX_THREAD: usize = 24;
const OVERHEAD: usize = 20 * 1024 * 1024;
const MAX_BYTES: usize = 4; // f32, u8 etc

fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();

    match args.command {
        Some(Command::Run {
            runs,
            warmups,
            max_threads,
        }) => {
            let available_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);

            let thread_num = if max_threads == 0 {
                available_threads
            } else {
                min(max_threads, min(available_threads, MAX_THREAD))
            };

            let runs = runs.clamp(1, MAX_RUN);
            let warmups = warmups.clamp(1, MAX_WARMUP);
            let stack_size = SAMPLE_SIZE * MAX_BYTES * POOL_CAPACITY + OVERHEAD;

            let main_handle = std::thread::Builder::new()
                .name("bench-thread".to_string())
                .stack_size(stack_size) // oh, yeah
                .spawn(move || {
                    let runtime =
                        tokio::runtime::Runtime::new().expect("Failed to create async runtime");
                    runtime.block_on(async {
                        run_benchmark(warmups, runs, thread_num)
                            .await
                            .expect("Benchmark failed");
                    });
                })?;
            main_handle.join().expect("Main thread panicked");
        }
        None => {
            print_ascii();
        }
    }
    Ok(())
}

fn print_ascii() {
    let ascii = "
          ▄▄                      ▄▄    
          ██                      ██    
    ▄████ ████▄ ▄█▀█▄ ████▄ ▄████ ████▄ 
    ██    ██ ██ ██▄█▀ ██ ██ ██    ██ ██ 
    ▀████ ████▀ ▀█▄▄▄ ██ ██ ▀████ ██ ██ 
    ";
    println!("{}", ascii);
    println!("Github: https://github.com/ronakgh97/cbench");
}
