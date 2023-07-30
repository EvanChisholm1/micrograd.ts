import { Value } from "../value";

const a = new Value(5);
console.log(a);
const b = a.mult(5);
console.log(b);
const c = new Value(2);
const d = b.add(c);
console.log(d);
const e = d.add(13).relu();
e.backward();
console.log(e);
