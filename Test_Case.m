function [ found ] = Test_Case( V, eigenfaces, indexes, i, j, mean_img )
    f = imread(['Faces/S' num2str(i) '/' num2str(indexes(i, j)) '.pgm']);
    [height, width] = size(f);
    f = double(reshape(f, [height * width, 1])) - mean_img;
    f = V' * f;
    [~, N] = size(eigenfaces);
    distance = Inf;
    found = 0;
    for k = 1 : N
        d = norm(double(f) - eigenfaces(:, k), 2);
        if distance > d
            found = k;
            distance = d;
        end
    end
end