#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>

// calculating the  derivative along x direction
double x_derivative(uint8_t right_pixel_val, uint8_t pixel_val)
{
	return double(int64_t(right_pixel_val) - int64_t(pixel_val));
}

// calculating the derivative along y direction
double y_derivative(uint8_t down_pixel_val, uint8_t pixel_val)
{
	return double(int64_t(down_pixel_val) - int64_t(pixel_val));
}

// calculating the corner response value of a position in img
// the response value is r=a*b-k*(a+b)^2, where a and b are matrix m's eigen values
// k usually set a value from 0.004 to 0.006
// return: the corner response value and two eigen values
std::tuple<double, double, double> corner_response(cv::Mat m, double k = 0.005)
{
	cv::Mat eigen_vals;
	cv::eigen(m, eigen_vals);
	double lambda1 = eigen_vals.at<double>(0), lambda2 = eigen_vals.at<double>(1);
	return std::make_tuple(lambda1 * lambda2 - k * pow(lambda1 + lambda2, 2), lambda1, lambda2);
}
// calculating at a pixel
// img: the image matrix
// window_size: the size of the window to calculating
// x,y: the position of current calculating pixel
std::tuple<double, double, double, bool> search_one_pos(cv::Mat img, int x, int y, int window_size = 3)
{
	int x_low = x - window_size / 2, y_low = y - window_size / 2,
		x_high = x + window_size / 2, y_high = y + window_size / 2;
	int img_width = img.size().width, img_height = img.size().height;
	if (x_low < 1 || y_low < 1 || x_high > img_width - 2 || y_high > img_height - 2)
		return std::make_tuple(0, 0, 0, false);
	cv::Mat m = cv::Mat(2, 2, CV_64F);
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			m.at<double>(i, j) = 0;
	for (int row = y_low; row <= y_high; row++)
	{
		for (int col = x_low; col <= x_high; col++)
		{
			m.at<double>(0, 0) += pow(x_derivative(uint8_t(*(img.ptr(row, col + 1))), uint8_t(*(img.ptr(row, col)))), 2);
			m.at<double>(0, 1) += x_derivative(uint8_t(*(img.ptr(row, col + 1))), uint8_t(*(img.ptr(row, col)))) * \
				y_derivative(uint8_t(*(img.ptr(row + 1, col))), uint8_t(*(img.ptr(row, col))));
			m.at<double>(1, 0) = m.at<double>(0, 1);
			m.at<double>(1, 1) += pow(y_derivative(uint8_t(*(img.ptr(row + 1, col))), uint8_t(*(img.ptr(row, col)))), 2);
		}
	}
	std::tuple<double, double, double> val = corner_response(m);
	return std::make_tuple(std::get<0>(val), std::get<1>(val), std::get<2>(val), true);
}
// generate the image to display the eigen max
void generate_imax_img(cv::Mat eigen_max)
{
	int width = eigen_max.size().width, height = eigen_max.size().height;
	cv::Mat eigen_max_img = cv::Mat(height, width, CV_8U);
	double min_val, max_val;
	cv::minMaxIdx(eigen_max, &min_val, &max_val);
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			*(eigen_max_img.ptr(i, j)) = (uint8_t)((eigen_max.at<double>(i, j) - min_val)
				/ (max_val - min_val) * 255);
		}
	}
	cv::namedWindow("eigen max");
	cv::imshow("eigen max", eigen_max_img);
	cv::waitKey(3000);
	// save as img
	cv::imwrite("./output/imax.png", eigen_max_img);
}

// generate the image to display the eigen mins
void generate_imin_img(cv::Mat eigen_min)
{
	int width = eigen_min.size().width, height = eigen_min.size().height;
	cv::Mat eigen_min_img = cv::Mat(height, width, CV_8U);
	double min_val, max_val;
	cv::minMaxIdx(eigen_min, &min_val, &max_val);
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			*(eigen_min_img.ptr(i, j)) = (uint8_t)((eigen_min.at<double>(i, j) - min_val)
				/ (max_val - min_val) * 255);
		}
	}
	cv::namedWindow("eigen min");
	cv::imshow("eigen min", eigen_min_img);
	cv::waitKey(3000);
	// save as image
	cv::imwrite("./output/imin.png", eigen_min_img);
}

// generate the image to display the corner response value
void generate_corner_res_img(cv::Mat corner_res)
{
	int width = corner_res.size().width, height = corner_res.size().height;
	double min_val, max_val;
	cv::minMaxIdx(corner_res, &min_val, &max_val);
	cv::Mat corner_response_img = cv::Mat(height, width, CV_8UC3);
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			corner_response_img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0,
				uint8_t((corner_res.at<double>(i, j) - min_val) / (max_val - min_val) * 255));
		}
	}
	cv::namedWindow("corner response");
	cv::imshow("corner response", corner_response_img);
	cv::waitKey(3000);
	// save as image
	cv::imwrite("./output/corner_response.png", corner_response_img);
}

// direct display corner response on raw image
// img: three channel rgb color image
// corner_res: the corner response value of each pixel, double type
void display_corner_res_on_raw(cv::Mat img, cv::Mat corner_res)
{
	int width = img.size().width, height = img.size().height;
	double min_val, max_val;
	cv::minMaxIdx(corner_res, &min_val, &max_val);
	cv::Mat out_img = cv::Mat(height, width, CV_8UC3);
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			// corner respone is positive and not very small, corner ,using red
			if (corner_res.at<double>(i, j) > 1)
			{
				out_img.at<cv::Vec3b>(i, j) = cv::Vec3b(uint8_t(*(img.ptr(i, j))),
					uint8_t(*(img.ptr(i, j) + 1)), uint8_t(corner_res.at<double>(i, j) / max_val * 255));
			}
			else if (corner_res.at<double>(i, j) < -1)
			{
				// corner response is negative and not very small, edge, using blue
				out_img.at<cv::Vec3b>(i, j) = cv::Vec3b(uint8_t(corner_res.at<double>(i, j) / min_val * 255),
					uint8_t(*(img.ptr(i, j) + 1)), uint8_t(*(img.ptr(i, j) + 2)));
			}
			else
			{
				out_img.at<cv::Vec3b>(i, j) = cv::Vec3b(uint8_t(*(img.ptr(i, j))), uint8_t(*(img.ptr(i, j) + 1)),
					uint8_t(*(img.ptr(i, j) + 2)));
			}
		}
	}
	// display and save image
	cv::namedWindow("");
	cv::imshow("", out_img);
	cv::waitKey(3000);
	cv::imwrite("./output/raw_on_response.png", out_img);
}

void search_in_img(cv::Mat img)
{
	int img_width = img.size().width, img_height = img.size().height;
	cv::Mat eigen_max = cv::Mat(img_height, img_width, CV_64F),
		eigen_min = cv::Mat(img_height, img_width, CV_64F),
		corner_res = cv::Mat(img_height, img_width, CV_64F);
	std::vector<std::tuple<int, int>> bounds_pos;
	for (int i = 0; i < img_height; ++i)
	{
		for (int j = 0; j < img_width; ++j)
		{
			std::tuple<double, double, double, bool> val = search_one_pos(img, j, i);
			if (std::get<3>(val) == false)
			{
				bounds_pos.push_back(std::make_tuple(i, j));
			}
			else
			{
				double cner_res = std::get<0>(val), e1 = std::get<1>(val), e2 = std::get<2>(val);
				eigen_max.at<double>(i, j) = std::max(e1, e2);
				eigen_min.at<double>(i, j) = std::min(e1, e2);
				corner_res.at<double>(i, j) = cner_res;
			}
		}
	}
	for (std::tuple<int, int> pos : bounds_pos)
	{
		eigen_max.at<double>(std::get<0>(pos), std::get<1>(pos)) = 0;
		eigen_min.at<double>(std::get<0>(pos), std::get<1>(pos)) = 0;
		corner_res.at<double>(std::get<0>(pos), std::get<1>(pos)) = 0;
	}
	// generate the corresponding images
	generate_imax_img(eigen_max);
	generate_imin_img(eigen_min);
	generate_corner_res_img(corner_res);
	display_corner_res_on_raw(img, corner_res);
}


int main()
{
	cv::Mat img = cv::imread(cv::String("img_lighter.png"));
	cv::resize(img, img, cv::Size(512,716));
	search_in_img(img);
	return 0;
}