# Micrograd.ts

Micrograd.ts is a simple autgradient/machine learning library written in typescript that has a very similar api to that of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Installation

Just copy and paste the value.ts file and the nn.ts file. I may make this an npm package or host it on a cdn at some point but for now copy and paste works.

## Features

Basic auto differentiation of compute graphs, basic Multi Layer Perceptron (MLP) class. For now you need to write your own training code, it's pretty simple to do so, just calculate loss and nudge params of network so loss decreases.

## Performance

Slow as hell. Don't use this for anything other than learning how basic Automatic differentiation works.

Maybe WebGPU support will come at some point but no promises.
