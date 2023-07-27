import { Value } from './value'

class Neuron {
    n: number;
    w: Value[];
    b: Value;

    constructor(n: number) {
        this.n = n;
        this.b = new Value(0); 
        this.w = new Array(n).fill(0).map(_ => new Value((Math.random() * 2) - 1))
    }

    getParams() {
        return [...this.w, this.b]
    }
    
    call(x: Value[]) {
        const out = this.w.reduce((acc, cur, i) => acc.add(cur.mult(x[i]), new Value(0))).add(this.b).relu();
        return out;
    }
}

class Layer {
    nNeurons: number;
    nInput: number;
    neurons: Neuron[];

    constructor(nNeurons: number, nInput: number) {
        this.nNeurons = nNeurons;
        this.nInput = nInput;
        this.neurons = new Array(nNeurons).fill(0).map(_ => new Neuron(nInput));
    }

    getParams() {
        return this.neurons.flatMap(x => x.getParams())
    }
    
    call(x: Value[]) {
        const out = this.neurons.map(ne => ne.call(x));
        return out;
    }
}

class MLP {
    layers: Layer[];
    constructor(nInput: number, nOutputs: number[]) {
        const size = [nInput, ...nOutputs]
        this.layers = new Array(nOutputs.length).fill(0).map((x, i) => new Layer(size[i], size[i + 1]));
    }

    call(x: Value[]) {
        const out = this.layers.reduce((prevOut, l) => l.call(prevOut), x);
        return out;
    }

    getParams() {
        return this.layers.flatMap(x => x.getParams());
    }
}

const net = new MLP(1, [5, 5, 1]);
console.log(net.call([new Value(10)])[0].value);

//const l = new Layer(10, 10);
//console.log(l.getParams());

//const ne = new Neuron(10);
//console.log(ne);
//console.log(ne.getParams());

