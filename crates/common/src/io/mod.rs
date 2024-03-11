mod edge_list;
mod md_tree_adj;
mod metis;
mod pace2023;

use clap::ValueEnum;
pub use edge_list::read_edge_list;
pub use md_tree_adj::read_md_tree_adj;
pub use md_tree_adj::write_md_tree_adj;
pub use metis::read_metis;
pub use metis::write_metis;
pub use pace2023::read_pace2023;

#[derive(Debug, Clone, Eq, PartialEq, ValueEnum)]
pub enum GraphFileType {
    Pace2023,
    Metis,
    EdgeList,
}
