mod linalg;
mod nn;
fn main() {}

#[cfg(test)]
mod tests {
    use crate::{linalg::*, nn};
    #[test]
    fn test_nn() {
        let mut nn = nn::NN::new(&[2, 5, 1]);
        nn.run(&Matrix::new(2, 1).randomize());
        println!("{:?}", nn.output());
    }

    // AI generated tests
    #[test]
    fn test_matrix() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.size(), (2, 3));
        let m = Matrix::from(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.size(), (2, 3));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(1, 2)], 6.0);

        let m = m.transpose();
        assert_eq!(m.size(), (3, 2));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 4.0);
        assert_eq!(m[(1, 0)], 2.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(2, 0)], 3.0);
        assert_eq!(m[(2, 1)], 6.0);

        let m = m.map(|x| x * 2.0);
        assert_eq!(m.size(), (3, 2));
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(0, 1)], 8.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 10.0);
        assert_eq!(m[(2, 0)], 6.0);

        let m = m.get_row(1);
        assert_eq!(m.size(), (1, 2));
        assert_eq!(m[(0, 0)], 4.0);
        assert_eq!(m[(0, 1)], 10.0);
        
        let m = Matrix::from(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let n = Matrix::from(2, 3, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let res = m.add(&n);
        assert_eq!(res.size(), (2, 3));
        assert_eq!(res[(0, 0)], 6.0);
        assert_eq!(res[(0, 1)], 8.0);
        assert_eq!(res[(0, 2)], 10.0);
        assert_eq!(res[(1, 0)], 12.0);
        assert_eq!(res[(1, 1)], 14.0);
        assert_eq!(res[(1, 2)], 16.0);


        let res = m.dot(&n.transpose());
        assert_eq!(res.size(), (2, 2));
        assert_eq!(res[(0, 0)], 38.0);
        assert_eq!(res[(0, 1)], 56.0);
        assert_eq!(res[(1, 0)], 92.0);
        assert_eq!(res[(1, 1)], 137.0);
    }
}
