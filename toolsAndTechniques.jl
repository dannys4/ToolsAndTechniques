cd("..\\ToolsAndTechniques");

using Pkg; Pkg.activate("ToolsTechniques");

Pkg.add(["DataFrames",
         "Images",
         "ImageMagick",
         "ImageBinarization",
         "MLJ",
         "Random",
         "StatsBase",
         "PrettyPrint",
         "CategoricalArrays",
         "Plots",
         "DecisionTree"]);

using DataFrames, Images, MLJ, Random, StatsBase, PrettyPrint,
      CategoricalArrays, Plots, ImageBinarization;

img1 = Images.load("binarize.jpg")

"""
    bernsen(img, rad, ij, contrast) -> neighborhood, pixel_val
Binarize (or threshhold) an image via local windows

# Arguments
    `img::Matrix{UInt8}`- The image to binarize
    `rad::Int`- The radius of the window
    `ij::CartesianIndex`- The row of the window's center pixel
    `contrast::UInt8`- The contrast level (Application-specific, default is 0x03)

# Returns
    `pix_val::UInt8`- 0x00 or 0xFF, which corresponds to whether the pixel is black or white

"""
function bernsen(img::Matrix{UInt8}, r::Int, ij::CartesianIndex{2}, c::UInt8=0x03)::Tuple{Vector,Bool}

    # Build an rxr neighborhood around (i,j)
    delta = CartesianIndices((-r:r, -r:r));
    nhood = img[ij .+ delta];
    # Get the minimum and maximum in that neighborhood
    Zmin,Zmax = extrema(nhood);
    # Make a local threshhold
    thresh = (Zmin+Zmax)/2;
    # If there is a large contrast
    if (Zmax - Zmin) >= c
        # Use the average of min and max as the threshhold
        return nhood[:], (img[ij] >= thresh);
    end
    # Otherwise, check if the average is high or low
    return nhood[:], (thresh >= 128);
end

function apply_bernsen(img::Matrix{UInt8}, r::Int, c::UInt8=0x02)
    idxs = CartesianIndices(img[r+1:end-r, r+1:end-r]);
    idxs_sz = size(idxs);
    result = Matrix(undef, idxs_sz[1], idxs_sz[2]);

    for ij in idxs
        result[ij] = bernsen(img, r, ij + CartesianIndex(r,r), c);
    end

    features = hcat([x[1] for x in result[:]]...)';
    img_bin = [x[2] for x in result];
    labels = img_bin[:];
    return features, labels, Matrix{Gray{Bool}}(img_bin);
end

img1_array = Matrix{UInt8}(reinterpret(UInt8, img1));
# Set radius equal to 5
rad = 5;

img1_old = img1[rad+1:end-rad, rad+1:end-rad];
img1_feat, img1_lab, img1_new = apply_bernsen(img1_array, rad);
border = zeros(Gray{Bool},(size(img1_old,1),5));
hcat(img1_old, border, img1_new, border, binarize(img1_old, Yen()))

white_idxs = findall(img1_lab);
black_idxs = setdiff(1:length(img1_lab), white_idxs);
n_resample = 10000;
white_resample = sample(white_idxs, n_resample);
black_resample = sample(black_idxs, n_resample);
img1_lab_resample = vcat(img1_lab[black_resample], img1_lab[white_resample]);
img1_feat_resample = vcat(img1_feat[black_resample,:], img1_feat[white_resample,:]);

perm_rows = shuffle(1:2*n_resample);
img1_feat = img1_feat_resample[perm_rows,:];
img1_lab = img1_lab_resample[perm_rows];

@load DecisionTreeClassifier
XG = @load XGBoostClassifier;

F = 5;
xgb = XG;
img1_feat_retype = coerce(img1_feat, autotype(img1_feat, :discrete_to_continuous));
m = machine(xgb, DataFrame(img1_feat_retype), CategoricalArray(img1_lab));
r = range(xgb, :num_round, lower=50, upper=500);
curve = learning_curve!(m, range=r, resolution=20,
                        measure=brier_score);
Plots.plot(curve.parameter_values, curve.measurements)
