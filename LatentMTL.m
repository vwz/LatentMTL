function [errdistBx errdistBy Jx Jy errdistB] = LatentMTL()
% Input: Device data, loaded from WiFiDeviceData.mat
% Output: errdistBx - average error distance for device B's x-coordinate 
%                     predictions at each iteration.
%         Jx - the objective function function value for x-coordinate at each iteration.
%         errdistBy & Jy - Similar to errdistBx & Jx, but for y-coordinate.
%         errdistB - average error distance over 2-D coordinates after convergence.

% Note: This code uses two auxilliary toolboxs.
%       (1) LIBSVM toolbox (matlab interface): libsvm-mat-2.85-1.
%           See http://www.csie.ntu.edu.tw/~cjlin/libsvm.
%       (2) Mosek toolbox (for solving quadratic programming).
%           See http://www.mosek.com/index.php?id=38.

% Copyright by Vincent W. Zheng (http://www.cse.ust.hk/~vincentz).
% Any question, please send email to vincentz_AT_cse.ust.hk.
% October 19th, 2008.
% ===============================================================

% Randomly split the data into training/test sets.
[nloc] = GetDeviceData();
load Data_all_device.mat;

% Common settings for multi-task learning
T = 2;
lambda1 = 1;
lambda2 = 0.5;
lambda3 = 0.1;
mu = T * lambda2 / lambda1;
C = T / (2 * lambda1);
pi1 = 1;
pi2 = 1;

% PCA low-dimensions
ndim = 60;

% Randomly get 10% of the labeled data from device B
nloc = 107;
a = randperm(nloc);
s = round(nloc * 0.1);
ref = a(1:s); % ref records the grids whose data are selected for training.

Y2trn_part = [];
X2trn_part = [];
P2trn_part = [];
for i=1:s
    index = find(Y2trn==ref(i));
    Y2trn_part = [Y2trn_part; Y2trn(index)];
    X2trn_part = [X2trn_part; X2trn(index,:)];
    P2trn_part = [P2trn_part; P2trn(index,:)];
end

% Maximum iterations
MaxIter = 10;
errdistBx = zeros(MaxIter,1);
errdistBy = zeros(MaxIter,1);
Jx = zeros(MaxIter,1);
Jy = zeros(MaxIter,1);


% *******************************************
disp('Consider the x-coordinate');

% Initialize \varphi_t by PCA
[eigvector, eigvalue] = PCA(X1trn, ndim);
varphi1 = eigvector';
[eigvector, eigvalue] = PCA(X2trn, ndim);
varphi2 = eigvector';

iter = 0;
while iter < MaxIter
    iter = iter + 1
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Step 1: Fix \varphi_t, optimize (w_0, v_t, \xi_it, \xi_it^*, b)
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    % Step 1.1: Map data to latent space
    
    X1trn_lat = X1trn * varphi1';
    X1tst_lat = X1tst * varphi1';
    X2trn_part_lat = X2trn_part * varphi2';
    X2tst_lat = X2tst * varphi2';
    
    
    % Step 1.2: Re-map data to fit a standard SVR
    
    [n1 d] = size(X1trn_lat);
    X1trn_map = [X1trn_lat/sqrt(mu) X1trn_lat zeros(n1,d)];
    [n1 d] = size(X1tst_lat);
    X1tst_map = [X1tst_lat/sqrt(mu) X1tst_lat zeros(n1,d)];
    [n2 d] = size(X2trn_part_lat);
    X2trn_map = [X2trn_part_lat/sqrt(mu) zeros(n2,d) X2trn_part_lat];
    [n2 d] = size(X2tst_lat);
    X2tst_map = [X2tst_lat/sqrt(mu) zeros(n2,d) X2tst_lat];

    clear X1trn_lat X1tst_lat X2trn_part_lat X2tst_lat;
    
    % Step 1.3: Train the standard SVR
    
    X = [X1trn_map; X2trn_map];
    n1 = size(X1trn_map,1);
    n2 = size(X2trn_map,1);

    % Consider x-coordinate
    Y = [P1trn(:,1); P2trn_part(:,1)];
    model = svmtrain(Y, X, '-s 3 -t 0');
    
    
    % Step 1.4: Derive the primal variables, using training data X: dec_values = X * w + b 
    
    % Recover b
    b = (-1) * model.rho;
    
    % Recover \xi_t and \xi_t^*
    [tstYx, accuracy, dec_values] = svmpredict(Y, X, model);
    e = 0.001; % default value in libsvm
    
    xi = (Y - dec_values) - e;
    index = find(xi<0);
    xi(index) = 0;
    xi1 = xi(1:n1);
    xi2 = xi(n1+1:end);
    
    xi_star = (dec_values - Y) - e;
    index = find(xi_star<0);
    xi_star(index) = 0;
    xi1_star = xi_star(1:n1);
    xi2_star = xi_star(n1+1:end);
    
    % Recover w_t
    w = pinv(X' * X) * X' * (dec_values - b);
    d = size(X,2) / 3;
    w0 = w(1:d) / sqrt(mu);
    v1 = w(d+1:2*d);
    w1 = w0 + v1;
    v2 = w(2*d+1:end);
    w2 = w0 + v2;


    % Step 1.5: Test the standard SVR on device B
    
    Y = P2tst(:,1);
    X = X2tst_map;
    [tstYx, accuracy, dec_values] = svmpredict(Y, X, model);
    P2pre = [tstYx P2tst(:,2)];

    % Calculate error distance
    temp = sum((P2pre-P2tst).^2,2);
    errdistBx(iter) = mean(sqrt(temp))
    
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Step 2: Calculate the J-value
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Jx(iter) = sum( pi1*(sum(xi1 + xi1_star)) + pi2*(sum(xi2 + xi2_star)) )...
        + lambda1 / T * (norm(v1)^2 + norm(v2)^2) ...
        + lambda2 * norm(w0)^2 ...
        + lambda3 / T * ( sum(sum(varphi1.^2)) + sum(sum(varphi2.^2)) )
    
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Step 3: Fix (w_0, v_t, \xi_it, \xi_it^*, b), optimize \varphi_t
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    % Step 3.1: For \varphi_1
    s = size(varphi1,1) * size(varphi1,2);
    varphi1_bar = reshape(varphi1, s, 1);
    
    B = kron(X1trn, w1');
    
    Y = P1trn(:,1);
    A1 = Y - b - (e + xi1);
    A2 = Y - b + (e + xi1_star);
    
    % ------------------------------------------------------
    % Solve the Quadratic Programming problem
    % syntex    : [res] = mskqpopt(q,c,a,blc,buc,blx,bux)
    %                   min 1/2 * x' * q * x + c' * x
    %                   s.t. blc <= a * x <= buc
    %                        blx <=   x   <= bux
    % ------------------------------------------------------
    q = eye(s);
    c = zeros(s,1);
    [res] = mskqpopt(q,c,B,A1,A2,[],[]);
    clear q c B A1 A2;
    varphi1_bar = res.sol.itr.xx;
    varphi1 = reshape(varphi1_bar, size(varphi1,1), size(varphi1,2));
    clear varphi1_bar;
    
    
    % Step 3.2: For \varphi_2
    s = size(varphi2,1) * size(varphi2,2);
    varphi2_bar = reshape(varphi2, s, 1);
    
    B = kron(X2trn_part, w2');
    
    Y = P2trn_part(:,1);
    A1 = Y - b - (e + xi2);
    A2 = Y - b + (e + xi2_star);
    
    % ------------------------------------------------------
    % Solve the Quadratic Programming problem
    % syntex    : [res] = mskqpopt(q,c,a,blc,buc,blx,bux)
    %                   min 1/2 * x' * q * x + c' * x
    %                   s.t. blc <= a * x <= buc
    %                        blx <=   x   <= bux
    % ------------------------------------------------------
    q = eye(s);
    c = zeros(s,1);
    [res] = mskqpopt(q,c,B,A1,A2,[],[]);
    clear q c B A1 A2;
    varphi2_bar = res.sol.itr.xx;
    varphi2 = reshape(varphi2_bar, size(varphi2,1), size(varphi2,2));
    clear varphi2_bar;
    
end

varphi1_x = varphi1;
varphi2_x = varphi2;


% *******************************************
disp('Consider the y-coordinate');

% Initialize \varphi_t by PCA
[eigvector, eigvalue] = PCA(X1trn, ndim);
varphi1 = eigvector';
[eigvector, eigvalue] = PCA(X2trn, ndim);
varphi2 = eigvector';

iter = 0;
while iter < MaxIter
    iter = iter + 1
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Step 1: Fix \varphi_t, optimize (w_0, v_t, \xi_it, \xi_it^*, b)
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    % Step 1.1: Map data to latent space
    
    X1trn_lat = X1trn * varphi1';
    X1tst_lat = X1tst * varphi1';
    X2trn_part_lat = X2trn_part * varphi2';
    X2tst_lat = X2tst * varphi2';
    
    
    % Step 1.2: Re-map data to fit a standard SVR
    
    [n1 d] = size(X1trn_lat);
    X1trn_map = [X1trn_lat/sqrt(mu) X1trn_lat zeros(n1,d)];
    [n1 d] = size(X1tst_lat);
    X1tst_map = [X1tst_lat/sqrt(mu) X1tst_lat zeros(n1,d)];
    [n2 d] = size(X2trn_part_lat);
    X2trn_map = [X2trn_part_lat/sqrt(mu) zeros(n2,d) X2trn_part_lat];
    [n2 d] = size(X2tst_lat);
    X2tst_map = [X2tst_lat/sqrt(mu) zeros(n2,d) X2tst_lat];

    clear X1trn_lat X1tst_lat X2trn_part_lat X2tst_lat;
    
    % Step 1.3: Train the standard SVR
    
    X = [X1trn_map; X2trn_map];
    n1 = size(X1trn_map,1);
    n2 = size(X2trn_map,1);

    % Consider y-coordinate
    Y = [P1trn(:,2); P2trn_part(:,2)];
    model = svmtrain(Y, X, '-s 3 -t 0');
    
    
    % Step 1.4: Derive the primal variables, using training data X: dec_values = X * w + b 
    
    % Recover b
    b = (-1) * model.rho;
    
    % Recover \xi_t and \xi_t^*
    [tstYx, accuracy, dec_values] = svmpredict(Y, X, model);
    e = 0.001; % default value in libsvm
    
    xi = (Y - dec_values) - e;
    index = find(xi<0);
    xi(index) = 0;
    xi1 = xi(1:n1);
    xi2 = xi(n1+1:end);
    
    xi_star = (dec_values - Y) - e;
    index = find(xi_star<0);
    xi_star(index) = 0;
    xi1_star = xi_star(1:n1);
    xi2_star = xi_star(n1+1:end);
    
    % Recover w_t
    w = pinv(X' * X) * X' * (dec_values - b);
    d = size(X,2) / 3;
    w0 = w(1:d) / sqrt(mu);
    v1 = w(d+1:2*d);
    w1 = w0 + v1;
    v2 = w(2*d+1:end);
    w2 = w0 + v2;


    % Step 1.5: Test the standard SVR on device B
    
    Y = P2tst(:,2);
    X = X2tst_map;
    [tstYx, accuracy, dec_values] = svmpredict(Y, X, model);
    P2pre = [P2tst(:,1) tstYx];

    % Calculate error distance
    temp = sum((P2pre-P2tst).^2,2);
    errdistBy(iter) = mean(sqrt(temp))
    
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Step 2: Calculate the J-value
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Jy(iter) = sum( pi1*(sum(xi1 + xi1_star)) + pi2*(sum(xi2 + xi2_star)) )...
        + lambda1 / T * (norm(v1)^2 + norm(v2)^2) ...
        + lambda2 * norm(w0)^2 ...
        + lambda3 / T * ( sum(sum(varphi1.^2)) + sum(sum(varphi2.^2)) )
    
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Step 3: Fix (w_0, v_t, \xi_it, \xi_it^*, b), optimize \varphi_t
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    % Step 3.1: For \varphi_1
    s = size(varphi1,1) * size(varphi1,2);
    varphi1_bar = reshape(varphi1, s, 1);
    
    B = kron(X1trn, w1');
    
    Y = P1trn(:,2);
    A1 = Y - b - (e + xi1);
    A2 = Y - b + (e + xi1_star);
    
    % ------------------------------------------------------
    % Solve the Quadratic Programming problem
    % syntex    : [res] = mskqpopt(q,c,a,blc,buc,blx,bux)
    %                   min 1/2 * x' * q * x + c' * x
    %                   s.t. blc <= a * x <= buc
    %                        blx <=   x   <= bux
    % ------------------------------------------------------
    q = eye(s);
    c = zeros(s,1);
    [res] = mskqpopt(q,c,B,A1,A2,[],[]);
    clear q c B A1 A2;
    varphi1_bar = res.sol.itr.xx;
    varphi1 = reshape(varphi1_bar, size(varphi1,1), size(varphi1,2));
    clear varphi1_bar;
    
    
    % Step 3.2: For \varphi_2
    s = size(varphi2,1) * size(varphi2,2);
    varphi2_bar = reshape(varphi2, s, 1);
    
    B = kron(X2trn_part, w2');
    
    Y = P2trn_part(:,2);
    A1 = Y - b - (e + xi2);
    A2 = Y - b + (e + xi2_star);
    
    % ------------------------------------------------------
    % Solve the Quadratic Programming problem
    % syntex    : [res] = mskqpopt(q,c,a,blc,buc,blx,bux)
    %                   min 1/2 * x' * q * x + c' * x
    %                   s.t. blc <= a * x <= buc
    %                        blx <=   x   <= bux
    % ------------------------------------------------------
    q = eye(s);
    c = zeros(s,1);
    [res] = mskqpopt(q,c,B,A1,A2,[],[]);
    clear q c B A1 A2;
    varphi2_bar = res.sol.itr.xx;
    varphi2 = reshape(varphi2_bar, size(varphi2,1), size(varphi2,2));
    clear varphi2_bar;
    
end

varphi1_y = varphi1;
varphi2_y = varphi2;


% ******************************************************
% Final test both x and y coordinates

% -----------------------------------
% Step 1.1: Map data to latent space

X1trn_lat = X1trn * varphi1_x';
X1tst_lat = X1tst * varphi1_x';
X2trn_part_lat = X2trn_part * varphi2_x';
X2tst_lat = X2tst * varphi2_x';


% Step 1.2: Re-map data to fit a standard SVR

[n1 d] = size(X1trn_lat);
X1trn_map = [X1trn_lat/sqrt(mu) X1trn_lat zeros(n1,d)];
[n1 d] = size(X1tst_lat);
X1tst_map = [X1tst_lat/sqrt(mu) X1tst_lat zeros(n1,d)];
[n2 d] = size(X2trn_part_lat);
X2trn_map = [X2trn_part_lat/sqrt(mu) zeros(n2,d) X2trn_part_lat];
[n2 d] = size(X2tst_lat);
X2tst_map = [X2tst_lat/sqrt(mu) zeros(n2,d) X2tst_lat];

clear X1trn_lat X1tst_lat X2trn_part_lat X2tst_lat;

% Step 1.3: Train the standard SVR

X = [X1trn_map; X2trn_map];
n1 = size(X1trn_map,1);
n2 = size(X2trn_map,1);

% Consider x-coordinate
Y = [P1trn(:,1); P2trn_part(:,1)];
model = svmtrain(Y, X, '-s 3 -t 0');


% Step 1.4: Test the standard SVR on device B

Y = P2tst(:,1);
X = X2tst_map;
[tstYx, accuracy, dec_values] = svmpredict(Y, X, model);
P2pre = tstYx;


% -----------------------------------
% Step 1.1: Map data to latent space

X1trn_lat = X1trn * varphi1_y';
X1tst_lat = X1tst * varphi1_y';
X2trn_part_lat = X2trn_part * varphi2_y';
X2tst_lat = X2tst * varphi2_y';


% Step 1.2: Re-map data to fit a standard SVR

[n1 d] = size(X1trn_lat);
X1trn_map = [X1trn_lat/sqrt(mu) X1trn_lat zeros(n1,d)];
[n1 d] = size(X1tst_lat);
X1tst_map = [X1tst_lat/sqrt(mu) X1tst_lat zeros(n1,d)];
[n2 d] = size(X2trn_part_lat);
X2trn_map = [X2trn_part_lat/sqrt(mu) zeros(n2,d) X2trn_part_lat];
[n2 d] = size(X2tst_lat);
X2tst_map = [X2tst_lat/sqrt(mu) zeros(n2,d) X2tst_lat];

clear X1trn_lat X1tst_lat X2trn_part_lat X2tst_lat;

% Step 1.3: Train the standard SVR

X = [X1trn_map; X2trn_map];
n1 = size(X1trn_map,1);
n2 = size(X2trn_map,1);

% Consider y-coordinate
Y = [P1trn(:,2); P2trn_part(:,2)];
model = svmtrain(Y, X, '-s 3 -t 0');


% Step 1.4: Test the standard SVR on device B

Y = P2tst(:,2);
X = X2tst_map;
[tstYx, accuracy, dec_values] = svmpredict(Y, X, model);
P2pre = [P2pre tstYx];


% Calculate error distance
temp = sum((P2pre-P2tst).^2,2);
errdistB = mean(sqrt(temp))



% ===============================================================
function [nloc] = GetDeviceData()

% Input: Device data, loaded from WiFiDeviceData.mat
%   e.g. (1) 'XA' denotes the data collected at device A, with rows as examples,
%            columns as features (signal strengths from 118 access points).
%        (2) 'LA' denotes the corresponding grid indices (label as grid) for XA.
%        (3) 'YA' denotes the corresponding 2-D coordinates (label as coordinate) for XA.
%        (4) Similarly, we have 'XB', 'LB', 'YB' for device B data.
% Output: Splitted training/test data for both devices, saved in Data_all_device.mat.
%         'nloc' is a dummy output here.

load WiFiDeviceData.mat;

% set parameters
nloc = 107;

% Set the number of training data and test data drawn from each grid as 10.
% That is, only 20 samples from each grid are used in the experiments.
snumtr = 10;
snumte = 10;

% derive device A as labeled training data
X1trn = [];
Y1trn = [];
X1tst = [];
Y1tst = [];
P1trn = [];
P1tst = [];
for i=1:nloc
    index = find(LA==i);
    a = randperm(length(index));
    index = index(a);
    X1trn = [X1trn; XA(index(1:snumtr),:)];
    Y1trn = [Y1trn; LA(index(1:snumtr))];
    P1trn = [P1trn; YA(index(1:snumtr),:)];
    X1tst = [X1tst; XA(index(snumtr+1:snumtr+snumte),:)];
    Y1tst = [Y1tst; LA(index(snumtr+1:snumtr+snumte))];
    P1tst = [P1tst; YA(index(snumtr+1:snumtr+snumte),:)];
end

% adding Gaussian noise N(0,0.1) for computation convenience.
[n d] = size(X1trn);
X1trn = X1trn + sqrt(0.01) * randn(n,d);
[n d] = size(X1tst);
X1tst = X1tst + sqrt(0.01) * randn(n,d);

% derive device B as labeled training/test data
X2trn = [];
Y2trn = [];
X2tst = [];
Y2tst = [];
P2trn = [];
P2tst = [];
for i=1:nloc
    index = find(LB==i);
    a = randperm(length(index));
    index = index(a);
    X2trn = [X2trn; XB(index(1:snumtr),:)];
    Y2trn = [Y2trn; LB(index(1:snumtr))];
    P2trn = [P2trn; YB(index(1:snumtr),:)];
    X2tst = [X2tst; XB(index(snumtr+1:snumtr+snumte),:)];
    Y2tst = [Y2tst; LB(index(snumtr+1:snumtr+snumte))];
    P2tst = [P2tst; YB(index(snumtr+1:snumtr+snumte),:)];
end

[n d] = size(X2trn);
X2trn = X2trn + sqrt(0.01) * randn(n,d);
[n d] = size(X2tst);
X2tst = X2tst + sqrt(0.01) * randn(n,d);

clear LA LB XA XB YA YB;
save('Data_all_device');










