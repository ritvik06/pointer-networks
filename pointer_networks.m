Multi_hot = readtable('convex_hull_multi_hot.csv');
One_hot = readtable('convex_hull_one_hot.csv');
Softmax = readtable('convex_hull_softmax.csv');

Multi_hot{:,1} = round(Multi_hot{:,1}/5+0.5);
Multi_hot = varfun(@mean,Multi_hot,'GroupingVariables','Var1')
One_hot{:,1} = round(One_hot{:,1}/5+0.5);
One_hot = varfun(@mean,One_hot,'GroupingVariables','Var1')
Softmax{:,1} = round(Softmax{:,1}/5+0.5);
Softmax = varfun(@mean,Softmax,'GroupingVariables','Var1')
figure;
hold on
axis([0,1000,0,1])
plot(Multi_hot{:,1}, Multi_hot.mean_test_accuracy, 'r');
plot(One_hot{:,1}, One_hot.mean_test_accuracy, 'b');
plot(Softmax{:,1}, Softmax.mean_test_accuracy,'g');
xlabel('Step (x10^3)');
ylabel('Accuracy');
legend('Multi-Ptr-Net', 'Hard-Ptr-Net', 'Softmax');
hold off