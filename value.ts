class Value {
    value: number;
    grad: number;
    previous: Value[]
    _backward: () => any;

    constructor(init: number, children: number[] = []) {
        this.value = init;
        this.grad = 0;
        this._backward = () => null;
        this.prev = children;
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

        return out;
    }
}

const a = new Value(5);
console.log(a)
const b = a.mult(5);
console.log(b);
const c = new Value(2)
const d = b.add(c)
console.log(d)
console.log(d.add(2))
