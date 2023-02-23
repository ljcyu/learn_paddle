let x = [2, 3, 6, 7,8,9,13,15,17];
let y = [7, 10, 19, 22,25,28,40,46,52];

let a=0
let b=0

for (let i=0;i<1000;i++){
  let da=0
  let db=0
  for(let j=0;j<x.length;j++){
    let z=a*x[j]+b
    //console.log(z)
    da+=(z-y[j])*x[j]
    db+=z-y[j]
  }
  //console.log(da,db)
  a-=0.01*da/x.length
  b-=0.01*db/x.length
}
console.log(a,b)