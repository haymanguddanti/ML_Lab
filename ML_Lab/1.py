probAbsentFriday=0.03
probFriday=0.2
# bayes Formula
# p(Absent|Friday)=p(Friday|Absent)p(Absent)/p(Friday)
# p(Friday|Absent)=p(Friday∩Absent)/p(Absent)
# Therefore the result is:
bayesResult=(probAbsentFriday/probFriday)
print(bayesResult * 100)