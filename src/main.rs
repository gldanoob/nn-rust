mod linalg;
mod nn;
fn main() {}

#[cfg(test)]
mod tests {
    use crate::{linalg::*, nn};
    #[test]
    fn test_nn() {
        let mut nn = nn::NN::new(&[2, 5, 1]);
        nn.run(&[1.0, 2.0]);
        println!("{:?}", nn.output());
    }
}
