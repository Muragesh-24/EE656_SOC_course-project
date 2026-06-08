% main_clustering.m
clc;
clear;

imageFileName = input('Enter the image file name (e.g., ''p2.jpg''): ', 's');
num_clusters = input('Enter the number of clusters (e.g., 5): ');

try
    img = imread(imageFileName);
    if size(img, 3) == 1
        img_rgb = cat(3, img, img, img);
    else
        img_rgb = img;
    end
    
    [rows, cols, ~] = size(img_rgb);
    img_reshaped = double(reshape(img_rgb, rows * cols, 3));
    
catch ME
    fprintf('Error reading image: %s\n', ME.message);
    return;
end

function displaySegmentedImage(original_img_reshaped, labels, original_dims, centroids, num_clusters, plot_title)
    centroids = double(round(max(0, min(255, centroids))));
    labels = max(1, min(num_clusters, round(labels)));
    
    segmented_image_data = centroids(labels, :);
    segmented_image_display = reshape(segmented_image_data, original_dims(1), original_dims(2), 3);
    
    figure;
    imshow(uint8(segmented_image_display));
    title(plot_title);
    drawnow;
end

fprintf('\n--- Running K-Means with %d clusters ---\n', num_clusters);
[idx_kmeans, C_kmeans] = kmeans(img_reshaped, num_clusters, 'Replicates', 5, 'MaxIter', 1000);
displaySegmentedImage(img_reshaped, idx_kmeans, [rows, cols], C_kmeans, num_clusters, ...
    sprintf('K-Means Segmentation (Clusters: %d)', num_clusters));
fprintf('K-Means clustering complete.\n');

fprintf('\n--- Running Fuzzy C-Means with %d clusters ---\n', num_clusters);
if ~license('test', 'Fuzzy_Logic_Toolbox')
    fprintf('Fuzzy Logic Toolbox not found. Skipping Fuzzy C-Means.\n');
else
    [centers_fcm, U_fcm] = fcm(img_reshaped, num_clusters);
    [~, idx_fcm] = max(U_fcm, [], 1); 
    idx_fcm = idx_fcm'; 
    displaySegmentedImage(img_reshaped, idx_fcm, [rows, cols], centers_fcm, num_clusters, ...
        sprintf('Fuzzy C-Means Segmentation (Clusters: %d)', num_clusters));
    fprintf('Fuzzy C-Means clustering complete.\n');
end

fprintf('\n--- Running Self-Optimal Clustering (SOC) with %d clusters ---\n', num_clusters);
try
    [GSS_soc, result_soc] = soc_implmt(img_rgb, num_clusters);
    fprintf('SOC Global Silhouette Score: %f\n', GSS_soc);
    
    centroids_soc = result_soc.cc_norm * 255; 
    if size(centroids_soc, 2) < 3
        centroids_soc = repmat(centroids_soc, 1, 3);
    end

    displaySegmentedImage(img_reshaped, result_soc.idx, [rows, cols], centroids_soc, num_clusters, ...
        sprintf('SOC Segmentation (Clusters: %d)', num_clusters));
    fprintf('Self-Optimal Clustering (SOC) complete.\n');

catch ME
    fprintf('Error running SOC: %s\n', ME.message);
    disp(ME.message);
end

fprintf('\n--- Running IMC1 with %d clusters ---\n', num_clusters);
figure;
imshow(img_rgb);
title(sprintf('IMC1 (Algorithm Undefined, Clusters: %d)', num_clusters));
drawnow;

fprintf('\n--- Running IMC2 with %d clusters ---\n', num_clusters);
figure;
imshow(img_rgb);
title(sprintf('IMC2 (Algorithm Undefined, Clusters: %d)', num_clusters));
drawnow;