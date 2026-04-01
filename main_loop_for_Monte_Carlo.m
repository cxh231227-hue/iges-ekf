Monte_carlo=100;
for iter=1:Monte_carlo
    if(mod(iter,10)==0)
        fprintf("---------Monte Carlo Iteration - normal %d---------------\n",iter)
    end
    evalc('dse_main_four_methods(1,0)');
end
Monte2=Monte_carlo/2;
for iter=1:Monte2
    if(mod(iter,10)==0)
        fprintf("---------Monte Carlo Iteration - big %d---------------\n",iter)
    end
   evalc('dse_main_four_methods(2,0)');
end
for iter=1:Monte_carlo
    if(mod(iter,10)==0)
        fprintf("---------Monte Carlo Iteration - small %d---------------\n",iter)
    end
    evalc('dse_main_four_methods(0.5,0)');
end