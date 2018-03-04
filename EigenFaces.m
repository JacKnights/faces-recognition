N = 7 * 40;
K = 90;
Test_Num = 3 * 40;
height = 112;
width = 92;

Test_Times = 100;
accuracy = zeros(100, Test_Times);
for K = 50 : 100
    for T = 1 : Test_Times
        indexes = zeros(N, 10);
        train_imgs = zeros(N, height, width);
        for i = 1 : 40
            indexes(i, :) = randperm(10);
            train_imgs((i - 1) * 7 + 1 : i * 7, :, :) = Get_Training_Set(['Faces/S' num2str(i)], indexes(i, 1 : 7), height, width, ['TrainSet/' num2str(i)]);
        end
        train_imgs = uint8(train_imgs);

        X = zeros(height * width, N);
        for i = 1 : N
            X(:, i) = reshape(train_imgs(i, :, :), [height * width, 1]);
        end
        mean_img = mean(X, 2);
        for i = 1: N
           X(:, i) = X(:, i) - mean_img; 
        end
        L = X' * X;
        [W, D] = eig(L);
        W = W(:, N - K + 1 : N);
        V = X * W;
        eigenfaces = V' * X;

        for i = 1 : 40
            for j = 8 : 10
                found = Test_Case(V, eigenfaces, indexes, i, j, mean_img);
                found = floor((found - 1) / 7) + 1;
                if found == i
                    accuracy(K, T) = accuracy(K, T) + 1;
                end
            end
        end
        accuracy(K, T) = double(accuracy(K, T) / (Test_Num));
    end
end
Mean_Accuracy = mean(accuracy, 2);
plot(Mean_Accuracy);
axis([50 100 0.9 1]);