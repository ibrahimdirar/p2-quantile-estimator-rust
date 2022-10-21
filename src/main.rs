use rand::distributions::Distribution;
use statrs::distribution::Normal;

mod p2_quantile {
    pub struct P2Algorithm {
        p: f64,
        count: usize,
        q: [f64; 5],
        n: [f64; 5],
        ns: [f64; 5],
    }

    impl P2Algorithm {
        pub fn new(p: f64) -> P2Algorithm {
            return P2Algorithm {
                p,
                count: 0,
                q: [0.0; 5],
                n: [0.0, 1.0, 2.0, 3.0, 4.0],
                ns: [0.0, 1.0, 2.0, 3.0, 4.0],
            };
        }
        pub fn observe(&mut self, value: f64) {
            self.count += 1;
            if self.count <= 5 {
                // add value to end of data.q
                self.q[self.count - 1] = value;
                if self.count == 5 {
                    // sort data.q
                    self.q.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
                return;
            }
            self.do_algo(value);
        }

        pub fn quantile(&self) -> f64 {
            return self.q[2];
        }

        fn do_algo(&mut self, value: f64) {
            let k: usize = self._find_cell(value);
            self.update_extremes(value);

            for i in (k + 1)..5 {
                self.n[i] += 1.0;
            }
            let count = self.count as f64;
            self.ns[1] = count * self.p / 2.0;
            self.ns[2] = count * self.p;
            self.ns[3] = count * (1.0 + self.p) / 2.0;
            self.ns[4] = count;

            self.update_marker_positions();
        }

        fn _find_cell(&mut self, value: f64) -> usize {
            let mut k = 3;
            for i in 0..4 {
                if self.q[i] <= value && value < self.q[i + 1] {
                    k = i;
                    break;
                }
            }
            return k;
        }

        fn update_extremes(&mut self, value: f64) {
            if value < self.q[0] {
                self.q[0] = value;
            }
            if value > self.q[4] {
                self.q[4] = value;
            }
        }

        fn update_marker_positions(&mut self) {
            for i in 1..4 {
                let d = self.ns[i] - self.n[i];
                let move_right = (d >= 1.0) && (self.n[i + 1] - self.n[i] > 1.0);
                let move_left = (d <= -1.0) && (self.n[i - 1] - self.n[i] < -1.0);
                if move_right || move_left {
                    let d: i8 = if d > 0.0 { 1 } else { -1 };
                    let qs = self.p2_interpolation(d, i);
                    if self.q[i - 1] < qs && qs < self.q[i + 1] {
                        self.q[i] = qs
                    } else {
                        self.q[i] = self.linear_interpolation(d, i)
                    }
                    self.n[i] += d as f64;
                }
            }
        }

        fn p2_interpolation(&mut self, d: i8, i: usize) -> f64 {
            let d = d as f64;
            return self.q[i]
                + d / (self.n[i + 1] - self.n[i - 1])
                    * ((self.n[i] - self.n[i - 1] + d) * (self.q[i + 1] - self.q[i])
                        / (self.n[i + 1] - self.n[i])
                        + (self.n[i + 1] - self.n[i] - d) * (self.q[i] - self.q[i - 1])
                            / (self.n[i] - self.n[i - 1]));
        }

        fn linear_interpolation(&mut self, d: i8, i: usize) -> f64 {
            let n = (i as i8 + d) as usize;
            return self.q[i] + d as f64 * (self.q[n] - self.q[i]) / (self.n[n] - self.n[i]);
        }
    }
}
fn main() {
    use std::time::Instant;
    // generate random data with normal distribution
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 10.0).unwrap();

    // observe some random data
    let mut all_observations_p2 = Vec::new();
    let mut all_observations_true = Vec::new();
    // collection of observations
    for _ in 0..10 {
        let mut observations = Vec::new();

        for _ in 0..10000001 {
            let value = normal.sample(&mut rng);
            observations.push(value);
        }
        all_observations_p2.push(observations.clone());
        all_observations_true.push(observations.clone());
    }
    let p2_now = Instant::now();
    for observations in all_observations_p2 {
        let now = Instant::now();
        let mut algorithm = p2_quantile::P2Algorithm::new(0.5);
        for value in observations {
            algorithm.observe(value);
        }
        let elapsed = now.elapsed();
        println!("P2 Median = {} in {:.2?}", algorithm.quantile(), elapsed);
    }
    let p2_elapsed = p2_now.elapsed();
    let true_now = Instant::now();
    for observations in all_observations_true {
        let now = Instant::now();
        let mut observations = observations;
        observations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = observations[observations.len() / 2];
        let elapsed = now.elapsed();
        println!("True Median = {} in {:.2?}", median, elapsed);
    }
    let true_elapsed = true_now.elapsed();
    println!();
    println!("P2 Median took {:.2?}", p2_elapsed);
    println!("True Median took {:.2?}", true_elapsed);
}
