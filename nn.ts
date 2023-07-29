import { Value } from './value'

class Neuron {
    n: number;
    w: Value[];
    b: Value;

    constructor(n: number) {
        this.n = n;
        this.b = new Value(0); 
        this.w = new Array(n).fill(0).map(_ => new Value((Math.random() * 2)))
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

const x = new Value(10);
const y = new Value(1);

const xs = new Array(1000).fill(0).map((_, i) => new Value(i));
const ys = xs.map(x  => x.value % 2 === 0 ? new Value(1) : new Value(0));

const STEP_SIZE = 0.0001;

for(let i = 0; i < 500; i++) {
    //const out = net.call([x])[0]
    //const loss = y.add(out.mult(-1));
    //let loss = new Value(0);
    xs.forEach((x, j) => {
        const out = net.call([x])[0];
        const loss = ys[j].add(out.mult(-1));
        //loss = loss.add(ys[i].add(out.mult(-1)));
        loss.grad = 1
        loss.backward();
        if(i % 100 === 0) console.log("x:", x.value);
        if(i % 100 === 0) console.log(loss.value);
            //console.log(out.value);

        for(const p of net.getParams()) {
            if(loss.value > 0) p.value -= p.grad * STEP_SIZE;
            if(loss.value < 0) p.value += p.grad * STEP_SIZE;
            p.grad = 0;
        }
    })

}

console.log("test data");
console.log("999", net.call([new Value(999)])[0].value);
console.log("998", net.call([new Value(998)])[0].value);
console.log("2", net.call([new Value(2)])[0].value);
console.log("1", net.call([new Value(1)])[0].value);
console.log("5000", net.call([new Value(5000)])[0].value);
console.log("-1", net.call([new Value(-1)])[0].value);

//const l = new Layer(10, 10);
//console.log(l.getParams());

//const ne = new Neuron(10);
//console.log(ne);
//console.log(ne.getParams());

