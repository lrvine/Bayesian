#! /usr/bin/perl -w

$infile = $ARGV[0];

$attribute = $ARGV[1];


@outname = split /\./ , $infile; 

$outfile = "$outname[0]_split.txt" ;


open(IN , "$infile");
open(OUT, ">$outfile");



@lines= <IN> ;
close(IN);

print OUT "($#lines+1) $attribute\n";


$dis = 0;
for ( $i=0 ; $i < $attribute  ; $i++){
     if($i == ($attribute-1) ){
       print OUT "$dis\n";
     }else{     
       print OUT "$dis ";
     }
}

$attributeclass = 10;
$ansclass = 2; 
for ( $i=0 ; $i < $attribute  ; $i++){
     print OUT "$attributeclass ";
}
print OUT "$ansclass\n";

for ( $i=0 ; $i <= $#lines  ; $i++)
{

  print " line $i \n";
  @values = split(',', $lines[$i]);
  for( $j=0 ; $j<= $#values ; $j++){
    if($j==0){
    }elsif( $j==$#values ){
      if($values[$j]==4){
        print OUT "2\n";
      }elsif( $values[$j]==2){
        print OUT "1\n";
      }
    }else{ 
      if($values[$j] =~/\?/){
        $t=5;
      }else{
        $t = $values[$j];
      }
      print OUT "$t ";
    }
  }
#  print OUT "\n";
}

 
print " job concluded !\n";

close(OUT)
