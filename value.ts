export class Value {
    value: number;
    grad: number;
    prev: Value[]
    _backward: () => any;
    visited: boolean = false;

    constructor(init: number, children: number[] = []) {
        this.value = init;
        this.grad = 0;
        this._backward = () => null;
        this.prev = [...children];
    }

    add(b: number | Value) {
        const other = typeof b === "object" ? b : new Value(b);
        const out = new Value(this.value + other.value, [this, other]);

        const _backward = () => {
            other.grad += out.grad;
            this.grad += out.grad;
        }

        out._backward = _backward;
        return out;
    }

    mult(b: number | Value) {
        const other = typeof b === "object" ? b : new Value(b);
        const out = new Value(this.value * other.value, [this, other]);

        const _backward = () => {
            other.grad += this.value * out.grad;
            this.grad += other.value * out.grad;
        }

        out._backward = _backward;
        return out;
    }

    relu() {
        const out = new Value(this.value > 0 ? this.value : 0, [this]);
        const _backward = () => {
            this.grad += (out.value > 0) * out.grad;
        }

        out._backward = _backward;
        return out;
    }

    backward() {
        const topo = [];
        const buildTopo = (s: Value) => {
            if(s.visited) return;

            s.visited = true;
            for(const c of s.prev) {
                buildTopo(c);
            }

            topo.push(s);
        }
        buildTopo(this);

        this.grad = 1;
        for(const s of topo.reverse()) {
            s._backward();
        }
    }
}

const a = new Value(5);
console.log(a)
const b = a.mult(5);
console.log(b);
const c = new Value(2)
const d = b.add(c)
console.log(d)
const e = d.add(13).relu()
e.backward();
console.log(e);

