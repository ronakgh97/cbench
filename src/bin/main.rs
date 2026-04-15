use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "cbench",
    version = "1.0.0-theta",
    about = "BLAS based cpu & gpu micro-benching tool"
)]
struct CliArgs {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Run {
        /// Number of runs to perform (default: 12)
        #[arg(short, long)]
        runs: Option<usize>,

        /// Number of warmup runs before benchmarking (default: 2)
        #[arg(short, long)]
        warmups: Option<usize>,

        /// Max threads to use in bench (default: all available thread)
        #[arg(short, long)]
        max_threads: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use cbench::prelude::*;
    let args = CliArgs::parse();

    match args.command {
        Some(Command::Run {
            runs,
            warmups,
            max_threads,
        }) => {
            let available = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1); // Default to single

            let thread_num = max_threads.unwrap_or(available);

            run_benchmark(runs, warmups, thread_num).await?;
        }
        None => {
            print_none();
        }
    }

    Ok(())
}

fn print_none() {
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
