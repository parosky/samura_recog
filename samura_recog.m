function [ds1, ds2] = samura_recog()
	img_size = [96, 96];
	gmm_dim = 512;
	pca_dim = 64;
	is_flip_pca = true;
	is_flip_gmm = true;
	directory = './img/';
	filename_pca = 'b*.jpg';
	filename_gmm = 'b*.jpg';

	% addpath(PATH_FOR_LIBSVM);
	% addpath(PATH_FOR_VLFEAT);
	vl_setup();

	pca = compute_pca(directory, filename_pca, img_size, pca_dim, is_flip_pca);
	gmm = compute_gmm(directory, filename_gmm, img_size, gmm_dim, is_flip_gmm, pca);

	ds1 = {};
	ds2 = {};
	for i=1:8
		filename = ['img/b', num2str(i), '.jpg'];
		ds1{i} = extract_descriptors(filename, img_size, false, pca, gmm);
		ds1{8+i} = extract_descriptors(filename, img_size, true, pca, gmm);
		filename = ['img/a', num2str(i), '.jpg'];
		ds2{i} = extract_descriptors(filename, img_size, false, pca, gmm);
		ds2{8+i} = extract_descriptors(filename, img_size, true, pca, gmm);
	end

	m = svmtrain([1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1]', cell2mat(ds1'), '-t 0 -b 1');
	[r1, r2, r3] = svmpredict([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]', cell2mat(ds2'), m, '-b 1');
	disp(r3(:,1)');
end

function img = read_image(filename, img_size)
	img = imread(filename);
	img = imresize(img, img_size);
	img = rgb2gray(img);
	img = im2single(img);
end

function descriptors = extract_descriptors(filename, img_size, is_flip, pca, gmm)
	factor = 2^(1/2);
	num_scales = 5;
	patchsize = 8;
	step = 1;

	% read image
	img = read_image(filename, img_size);
	if (nargin >= 3) && (is_flip)
		img = fliplr(img);
	end

	% extract descriptors
	descriptors = {};
	frames = {};
	for i=1:num_scales
		[f,d] = vl_dsift(img, 'Step', step, 'Size', patchsize/2*factor^(i-1));
		frames{i} = double(f');
		descriptors{i} = double(d');
	end
	frames = cell2mat(frames');
	descriptors = cell2mat(descriptors');

	% apply PCA
	if nargin >= 4
		descriptors = descriptors - repmat(pca.mean, size(descriptors, 1), 1);
		descriptors = descriptors * pca.coeff;
		descriptors = descriptors(:,1:pca.dim);
		descriptors = [descriptors, (frames ./ repmat(img_size, size(frames, 1), 1) -1/2)];
	end

	% compute FV
	if nargin >= 5
		descriptors = vl_fisher(descriptors', gmm.means, gmm.covariances, gmm.priors, 'Improved')';
	end
end

function pca = compute_pca(directory, filename, img_size, pca_dim, is_flip)
	filelist = dir([directory, filename]);

	descriptors = {};
	for i=1:size(filelist,1 )
		fn = [directory, filelist(i).name];
		descriptors{i} = extract_descriptors(fn, img_size);
		if is_flip
			descriptors{size(filelist,1)+i} = extract_descriptors(fn, img_size, true);
		end
	end
	descriptors = cell2mat(descriptors');

	pca = {};
	pca.coeff = princomp(descriptors);
	pca.mean = mean(descriptors);
	pca.dim = pca_dim;
end

function gmm = compute_gmm(directory, filename, img_size, gmm_dim, is_flip, pca)
	filelist = dir([directory, filename]);

	descriptors = {};
	for i=1:size(filelist,1 )
		fn = [directory, filelist(i).name];
		descriptors{i} = extract_descriptors(fn, img_size, false, pca);
		if is_flip
			descriptors{size(filelist,1)+i} = extract_descriptors(fn, img_size, true, pca);
		end
	end
	descriptors = cell2mat(descriptors');

	gmm = {};
	[gmm.means, gmm.covariances, gmm.priors] = vl_gmm(descriptors', gmm_dim, 'verbose');
end
