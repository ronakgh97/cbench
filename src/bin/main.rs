#[allow(unused)]
use cbench::bencher::{SAMPLE_SIZE, run_benchmark};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "cbench",
    version = "1.0.0-alpha",
    about = "BLAS based cpu micro-benching tool"
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
                std::cmp::min(max_threads, available_threads)
            };

            let main_handle = std::thread::Builder::new()
                .name("bench-thread".to_string())
                .stack_size(512 * 1024 * 1024)
                .spawn(move || {
                    let runtime = tokio::runtime::Runtime::new().unwrap();
                    runtime.block_on(async {
                        run_benchmark(runs, warmups, thread_num)
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
