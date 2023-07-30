export class Value {
    value: number;
    grad: number;
    prev: Value[];
    _backward: () => any;
    visited: boolean = false;

    constructor(init: number, children: Value[] = []) {
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
        };

        out._backward = _backward;
        return out;
    }

    mult(b: number | Value) {
        const other = b instanceof Value ? b : new Value(b);
        const out = new Value(this.value * other.value, [this, other]);

        const _backward = () => {
            other.grad += this.value * out.grad;
            this.grad += other.value * out.grad;
        };

        out._backward = _backward;
        return out;
    }

    sub(b: number | Value) {
        const other = b instanceof Value ? b : new Value(b);
        const out = this.add(other.mult(-1));

        return out;
    }

    relu() {
        const out = new Value(this.value > 0 ? this.value : 0, [this]);
        const _backward = () => {
            this.grad += (out.value > 0 ? 1 : 0) * out.grad;
        };

        out._backward = _backward;
        return out;
    }

    backward() {
        const topo: Value[] = [];
        const buildTopo = (s: Value) => {
            if (s.visited) return;

            s.visited = true;
            for (const c of s.prev) {
                buildTopo(c);
            }

            topo.push(s);
        };
        buildTopo(this);

        this.grad = 1;
        for (const s of topo.reverse()) {
            s._backward();
            s.visited = false;
        }
    }
}
