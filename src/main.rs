use burn::{
    backend::NdArray,
    data::{
        dataloader::batcher::Batcher,
    },
    prelude::*,
};

pub const INPUT_SIZE : usize = 3;

pub struct DataSample {
    pub input:[f32;INPUT_SIZE],//put actual size here
    pub target:f32,//assuming simple regression
}

#[derive(Clone, Debug)]
pub struct MyBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}


#[derive(Clone,Copy, Debug, Default)]
pub struct MyBatcher;

impl<B: Backend> Batcher<B, DataSample, MyBatch<B>> for MyBatcher {
    fn batch(&self, items: Vec<DataSample>, device: &B::Device) -> MyBatch<B> {
        let targets  = items
            .iter()
            .map(|sample|{
                Tensor::<NdArray,1>::from([sample.target])
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

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch_size, _] = input.dims();

        let x = self.l1.forward(input);
        let x = self.activation.forward(x);

        let x = self.l2.forward(x);
        let x = self.activation.forward(x);
        x.reshape([batch_size])
    }

    // pub fn forward_classification(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
    //     let targets = item.targets;
    //     let output = self.forward(item.images);
    //     let loss = CrossEntropyLossConfig::new()
    //         .init(&output.device())
    //         .forward(output.clone(), targets.clone());

    //     ClassificationOutput {
    //         loss,
    //         output,
    //         targets,
    //     }
    // }
}

fn main() {
    println!("Hello, world!");
}
