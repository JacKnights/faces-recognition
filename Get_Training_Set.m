function [ imgs ] = Get_Training_Set( input_path, index, height, width, output_path )
    imgs = zeros(length(index), height, width);
    for i = 1 : length(index)
        imgs(i, :, :) = uint8(imread([input_path '/' num2str(index(i)) '.pgm']));
    end
end