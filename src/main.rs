use burn::record::CompactRecorder;
use std::time::Instant;
use burn::data::dataloader::DataLoader;
use burn::backend::Autodiff;
use burn::train::StoppingCondition;
use burn::train::metric::store::Split;
use burn::train::metric::store::Direction;
use burn::train::metric::store::Aggregate;
use burn::train::MetricEarlyStoppingStrategy;
use burn::train::metric::LearningRateMetric;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::composed::ComposedLrSchedulerConfig;
use burn::data::dataset::Dataset;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamWConfig;
use burn::train::TestStep;
use burn::train::ValidStep;
use burn::train::TrainOutput;
use burn::tensor::backend::AutodiffBackend;
use burn::train::TrainStep;
use burn::tensor::linalg::l2_norm;
use burn::train::RegressionOutput;

use burn::{
    backend::{NdArray},
    data::{
        dataloader::batcher::Batcher,
    },
    prelude::*,
};



pub const INPUT_SIZE : usize = 3;
pub const BATCH_SIZE : usize = 32;
pub const ARTIFACT_DIR:&'static str = "training_output";

#[derive(Debug,Clone)]
pub struct DataSample {
    pub input:[f32;INPUT_SIZE],//put actual size here
    pub target:f32,//assuming simple regression
}

#[derive(Clone, Debug)]
pub struct MyBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,//this is actually [batch,1]
}


#[derive(Clone,Copy, Debug, Default)]
pub struct MyBatcher;

impl<B: Backend> Batcher<B, DataSample, MyBatch<B>> for MyBatcher {
    fn batch(&self, items: Vec<DataSample>, device: &B::Device) -> MyBatch<B> {
        let targets  = items
            .iter()
            .map(|sample|{
                Tensor::<NdArray,2>::from([[sample.target]])
            })
            .collect();
        let targets = Tensor::cat(targets, 0);
        let targets = Tensor::from_data(targets.into_data(),device);

        let inputs  = items
            .iter()
            .map(|sample|{
                Tensor::<NdArray,2>::from([sample.input])
            })
            .collect();

        let inputs = Tensor::cat(inputs, 0);
        let inputs = Tensor::from_data(inputs.into_data(),device);


        MyBatch{inputs,targets}
    }
}


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    activation: nn::Gelu,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let l1 = nn::LinearConfig::new(INPUT_SIZE, 128).init(device);
        let l2 = nn::LinearConfig::new(128, 1).init(device);
        Self{
            l1,l2,
            activation:nn::Gelu::new()
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // let [batch_size, _] = input.dims();

        let x = self.l1.forward(input);
        let x = self.activation.forward(x);

        let x = self.l2.forward(x);
        let x = self.activation.forward(x);
        // x.reshape([batch_size])
        x
    }

    pub fn forward_regress(&self, item: MyBatch<B>) -> RegressionOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.inputs);
        let loss = l2_norm(targets.clone()-output.clone(),1);
        
        let [batch_size, o] = loss.dims();
        assert_eq!(o,1);
        let loss = loss.reshape([batch_size]);
        
        RegressionOutput{
            loss,
            output,targets
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MyBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: MyBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regress(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MyBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: MyBatch<B>) -> RegressionOutput<B> {
        self.forward_regress(item)
    }
}

impl<B: Backend> TestStep<MyBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: MyBatch<B>) -> RegressionOutput<B> {
        self.forward_regress(item)
    }
}

pub struct MyDataset<T>(pub Vec<T>);
impl<T:Send+Sync+Clone> Dataset<T> for MyDataset<T>{
fn get(&self, i: usize) -> Option<T> { self.0.get(i).cloned() }
fn len(&self) -> usize { self.0.len() }
}

pub fn run_train<B:AutodiffBackend>(device:&B::Device){
    let model: Model<B> = Model::new(device);


    let train_data = vec![
        DataSample { input: [1.0, 2.0, 3.0], target:  6.0 },
        DataSample { input: [2.0, 3.0, 4.0], target:  9.0 },
        DataSample { input: [3.0, 4.0, 5.0], target: 12.0 },
        DataSample { input: [4.0, 5.0, 6.0], target: 15.0 },
    ];

    let valid_data = vec![
        DataSample { input: [10.0, 10.0, 10.0], target: 30.0 },
        DataSample { input: [ 1.0,  1.0,  1.0], target:  3.0 },
    ];

    let config_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(5e-5);


    let dataloader_train = DataLoaderBuilder::new(MyBatcher::default())
        .batch_size(BATCH_SIZE)
        .shuffle(23/*my seed*/)
        .num_workers(4/*put something sensible here*/)
        .build(MyDataset(train_data));

    let dataloader_valid = DataLoaderBuilder::new(MyBatcher::default())
        .batch_size(BATCH_SIZE)
        .shuffle(23/*my seed*/)
        .num_workers(4)
        .build(MyDataset(valid_data));

    let lr_scheduler = ComposedLrSchedulerConfig::new()
        .cosine(CosineAnnealingLrSchedulerConfig::new(1.0, 2000))
        // Warmup
        .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 2000))
        .linear(LinearLrSchedulerConfig::new(1e-2, 1e-6, 10000));


    //make a fresh result dir
    std::fs::remove_dir_all(ARTIFACT_DIR).unwrap();
    std::fs::create_dir_all(ARTIFACT_DIR).unwrap();

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metrics((LossMetric::new(), LossMetric::new()))
        .metric_train_numeric(LearningRateMetric::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 5 },
        ))
        .num_epochs(4)
        .summary()
        .build(
            model,
            config_optimizer.init(),
            lr_scheduler.init().unwrap()
        );

    let now = Instant::now();
    let result = learner.fit(dataloader_train, dataloader_valid);
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    result
        .model
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

}


fn main() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    let device = WgpuDevice::default();
    run_train::<Autodiff<Wgpu>>(&device);
}
