/*js版的拟合曲线*/
let x = [2, 3, 6, 7,8,9,13,15,17];
let y = [7, 10, 19, 22,25,28,40,46,52];

let a = 0
let b = 0
for (let i = 0; i < 10000; i++) {
    da = 0;
    db = 0;
    for (let j = 0; j < x.length; j++) {
        dz = y[j] - (a * x[j] + b)
        da += x[j] * dz
        db += dz
    }
    console.log(i+"-->"+da+' '+db+' '+dz);
    da = da/x.length;
    db = db/x.length;
    a += 0.001*da;
    b += 0.001*db;
    console.log(i+"-->"+a+' '+b);
}
console.log(a + " " + b);