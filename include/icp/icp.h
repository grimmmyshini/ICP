#include <iostream>
#include <time.h>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <Eigen/Core>
#include <Eigen/SVD>

#ifdef DEBUG_BUILD 
    #define DEBUG(msg) std::cerr << msg << std::endl
#else 
    #define DEBUG(msg) do {} while(0) 
#endif

struct correspondence{

    int src_index, tgt_index;
    float distance_sq;

    correspondence() : src_index(0), tgt_index(0), distance_sq(0){}
};

class IterativeClosestPoint{

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_; 
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_; 
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_;
    std::vector<int> src_indices_, tgt_indices_; 
    std::vector<struct correspondence> corr_;
    bool converged_ = false;
    Eigen::Matrix<float, 4, 4> final_transformation_; 
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree_;

    public:

    IterativeClosestPoint() : input_(new pcl::PointCloud<pcl::PointXYZ>), target_(new pcl::PointCloud<pcl::PointXYZ>),
     output_(new pcl::PointCloud<pcl::PointXYZ>), corr_(),
     src_indices_(), tgt_indices_(), tree_(new  pcl::KdTreeFLANN<pcl::PointXYZ>),
     final_transformation_(Eigen::Matrix<float, 4, 4>::Identity()) {}

    ~IterativeClosestPoint() {};

    void setInput(pcl::PointCloud<pcl::PointXYZ>::Ptr in){

        input_ = in;
    }

    void setTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr in){

        target_ = in;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr getInput(){

        return input_;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr getTarget(){

        return target_;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr getOutput(){

        return output_;
    }

    void setInputIndices(const std::vector<int>& indices){

        src_indices_ = indices; 
    }

    void setTargetIndices(const std::vector<int>& indices){

        tgt_indices_ = indices; 
    }

    void getInputIndices(std::vector<int>& indices){

        indices = src_indices_;
    }

    void getTargetIndices(std::vector<int>& indices){

        indices = tgt_indices_;
    }

    void correspondenceEstimation(double max_distance = std::numeric_limits<double>::max ());

    void processPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);

    void normaliseCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr normalised_src, pcl::PointCloud<pcl::PointXYZ>::Ptr normalised_tgt,
    Eigen::Vector3f mean_src, Eigen::Vector3f mean_tgt);

    Eigen::Matrix4f SVD(pcl::PointCloud<pcl::PointXYZ>::Ptr normalised_src, pcl::PointCloud<pcl::PointXYZ>::Ptr normalised_tgt, 
    Eigen::Vector3f mean_src, Eigen::Vector3f mean_tgt);
    
    bool checkConvergence(Eigen::Matrix4f trans);

    void align(pcl::PointCloud<pcl::PointXYZ>::Ptr final_out);
};
 

void IterativeClosestPoint::correspondenceEstimation(double max_distance){


    clock_t t = clock();

    //Assume every point has a valid correspondence 
    corr_.resize(src_indices_.size());


    std::vector<int> index(1);
    std::vector<float> distance(1);
    
    //total valid correspondences after estimation 
    int total_correspondences = 0;

    //set input to kd tree as the cloud that we have to match 
    tree_->setInputCloud(target_);

    //iterate to find the first nearest neighbour of all points in the point cloud
    for(std::vector<int>::const_iterator it = src_indices_.begin(); it != src_indices_.end(); it++){

      tree_->nearestKSearch (input_->points[*it], 1, index, distance);
      if (distance[0] > max_distance * max_distance)
        continue;
      
      //set all correspondence values 
      corr_[total_correspondences].src_index = *it;
      corr_[total_correspondences].tgt_index = index[0];
      corr_[total_correspondences].distance_sq = distance[0];
      total_correspondences++;
    }

    corr_.resize(total_correspondences);

    t = clock() - t;
    DEBUG( "Total correspondence estimation time is " + std::to_string( ( static_cast<float>(t)/CLOCKS_PER_SEC )*1000 ) + " ms");

}

void IterativeClosestPoint::processPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>());

    //remove outliers 
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> stat_ftr;
    stat_ftr.setInputCloud(cloud_in);
    stat_ftr.setMeanK(10);
    stat_ftr.setStddevMulThresh(0.2);
    stat_ftr.filter(*cloud_transformed);

    //filter data
    pcl::VoxelGrid<pcl::PointXYZ> fltr;
    fltr.setInputCloud(cloud_transformed);
    fltr.setLeafSize (0.01f, 0.01f, 0.01f);
    fltr.filter (*cloud_transformed);

    cloud_in = cloud_transformed;

}

//TO-DO: Convert to vector ops
void IterativeClosestPoint::normaliseCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr norm_src, pcl::PointCloud<pcl::PointXYZ>::Ptr norm_tgt, Eigen::Vector3f mean_src, Eigen::Vector3f mean_tgt){

    pcl::PointCloud<pcl::PointXYZ>::PointType centroid_src, centroid_tgt;

    //copy input and target to prepare base for point clouds to be normalised
    copyPointCloud(*input_, *norm_src);
    copyPointCloud(*target_, *norm_tgt); 
    int size = corr_.size();

    //************************************
    //calculate the centroid(mean) of the matched points by
    //*     do mean += some_point(i) for all i, then
    //*     mean /= some_point.size() 
    //************************************
    for(std::vector<correspondence>::iterator it = corr_.begin(); it != corr_.end(); it++){

        centroid_src.x += norm_src->points[it->src_index].x; 
        centroid_src.y += norm_src->points[it->src_index].y; 
        centroid_src.z += norm_src->points[it->src_index].z;

        centroid_tgt.x += norm_tgt->points[it->tgt_index].x;
        centroid_tgt.y += norm_tgt->points[it->tgt_index].y;
        centroid_tgt.z += norm_tgt->points[it->tgt_index].z;
        
    }

    centroid_src.x /= (size + 1); 
    centroid_src.y /= (size + 1); 
    centroid_src.z /= (size + 1);

    centroid_tgt.x /= (size + 1);
    centroid_tgt.y /= (size + 1);
    centroid_tgt.z /= (size + 1);

    int i = 0, j =0;    

    //shift all the points in the point cloud to form the normalized cloud 
    //TO-DO only shift the matched correspondences 
    while( i < norm_src->size() || j < norm_tgt->size()){

        if( i < norm_src->size() ){
            
            norm_src->points[i].x -= centroid_src.x;
            norm_src->points[i].y -= centroid_src.y;
            norm_src->points[i].z -= centroid_src.z;
            i++;
        }

        if( j < norm_tgt->size() ){
            
            norm_tgt->points[i].x -= centroid_tgt.x;
            norm_tgt->points[i].y -= centroid_tgt.y;
            norm_tgt->points[i].z -= centroid_tgt.z;
            j++;
        }
    }

    mean_src = Eigen::Vector3f(centroid_src);
    mean_tgt = Eigen::Vector3f(centroid_tgt);

}

Eigen::Matrix4f IterativeClosestPoint::SVD(pcl::PointCloud<pcl::PointXYZ>::Ptr norm_src, pcl::PointCloud<pcl::PointXYZ>::Ptr norm_tgt, Eigen::Vector3f mean_src, Eigen::Vector3f mean_tgt){

    Eigen::Matrix3f M = Eigen::Matrix3f::Constant(3, 3, 0);
    
    for(auto it = corr_.begin(); it != corr_.end(); it++){

        Eigen::Vector3f src_p(norm_src->points[it->src_index]), tgt_p(norm_tgt->points[it->tgt_index]); 
        M += ( src_p*(tgt_p.transpose()) ); 

    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3f result_rotation = (svd.matrixV()*(svd.matrixU().transpose())).transpose();
    Eigen::Matrix<float, 1, 3>  result_translation = ( mean_src - (mean_tgt * result_rotation) );
    
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();

    for(int i = 0; i < 3; i++) 
        transformation(3, i) = result_translation(0, i);
    
    for(int i=0; i < 3; i++ )
        for(int j=0; j < 3; j++)
            transformation(i, j) = result_rotation(i, j);

    return transformation;
    
}

bool IterativeClosestPoint::checkConvergence(Eigen::Matrix4f transformation){

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_tgt;
    double mean_sq_error = 0;

    pcl::transformPointCloud (*target_, *transformed_tgt, transformation);

    for(auto it = corr_.begin(); it != corr_.end(); it++)
        mean_sq_error += it->distance_sq;

    mean_sq_error /= corr_.size();

    if( mean_sq_error < 12e-4 ){

        converged_ = true;
        final_transformation_ = transformation;
        return true;
    
    }

    target_ = transformed_tgt;
    return false;
    
}