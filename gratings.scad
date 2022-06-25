//textures for the mice maze

plateX = 50;
plateY = 50;
plateZ = 3;

tol = 0.1;





module texture1(){
    for (i = [5:8: plateX-9]){
        translate([i, 0, 0]){
            cube([5,plateY,plateZ+3]);
        }//end translate
    }//end for 
    
    
    
    }
translate([110,0,0]){
cube([plateX,plateY,plateZ]);    
texture1();
}


cube([plateX,plateY,plateZ]);    

rotate([0,0,90]){
translate([0,-50,0]){
texture1();
}
}