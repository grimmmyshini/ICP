#include <pcl/filters/filter.h>
#include "include/icp/icp.h"

int
 main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>(5,1));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ> output;

  // Fill in the CloudIn data
  for (auto& point : *cloud_in)
  {
    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
  }
  
  std::cout << "Saved " << cloud_in->points.size () << " data points to input:" << std::endl;
      
  for (auto& point : *cloud_in)
    std::cout << point << std::endl;
      
  *cloud_out = *cloud_in;
  
  std::cout << "size:" << cloud_out->points.size() << std::endl;
  for (auto& point : *cloud_out){
    point.x += 0.7f;
    point.y += 0.2f;
   }

  std::cout << "Transformed " << cloud_in->points.size () << " data points:" << std::endl;
      
  for (auto& point : *cloud_out)
    std::cout << point << std::endl;

// PREPROCESS DATA

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud_in, *cloud_out, indices);

  if(output.points.size() != cloud_in->points.size())
    output.points.resize(cloud_in->points.size());
  
  output.header = cloud_in->header;
  output.width =  static_cast<uint32_t>(cloud_in->width);
  output.height = cloud_in->height; 

  for (size_t i = 0; i < indices.size (); ++i)
    output.points[i] = cloud_in->points[indices[i]];

  bool converged =  false; 
  Eigen::Matrix<float, 4, 4> transformation_matrix, prev_transformation;
  transformation_matrix = prev_transformation = Eigen::Matrix<float, 4, 4>::Identity(); 

  //to aid rigid transform (as per pcl implementation)
  for (size_t i = 0; i < indices.size(); ++i)
    output.points[i].data[3] = 1.0;
 
  IterativeClosestPoint icp;

  icp.setInput(cloud_in);
  icp.setTarget(cloud_out);
  icp.setInputIndices(indices);
  icp.setTargetIndices(indices);

 return (0);
}