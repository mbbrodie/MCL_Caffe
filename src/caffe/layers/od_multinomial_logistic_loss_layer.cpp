#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void ODMultinomialLogisticLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	this->assign_counts_full_.Reshape(bottom[0]->num(), 1, 1, 1);
}


//Multinomial Loss for 'Optimal Dieting'
template <typename Dtype>
void ODMultinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	this->best_pred_.Reshape(bottom[0]->num(), this->layer_param_.mcl_param().hard_k(), 1, 1);

	this->assign_counts_.Reshape(bottom.size()-1, 1, 1, 1);
	top[0]->Reshape(bottom.size()-1, 1, 1, 1);
	if(top.size() >= 2) {
		top[1]->Reshape(bottom.size()-1,1,1,1);
		top[2]->Reshape(bottom.size()-1,1,1,1);
	}
	CHECK_EQ(bottom[bottom.size()-1]->channels(), 1);
	CHECK_EQ(bottom[bottom.size()-1]->height(), 1);
	CHECK_EQ(bottom[bottom.size()-1]->width(), 1);
}



template <typename Dtype>
void ODMultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int n_pred = bottom.size()-1;

  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int k =  this->layer_param_.mcl_param().hard_k();
  Dtype* best = this->best_pred_.mutable_cpu_data();
  Dtype* counts = this->assign_counts_.mutable_cpu_data();
  Dtype* losses = top[0]->mutable_cpu_data();
  //Dtype* ac_full = this->assign_counts_full_.mutable_cpu_data();
  
  caffe_set(n_pred, Dtype(0), counts);
  //caffe_set(n_pred, *this->assign_counts_full_.mutable_cpu_data(), ac_full);
  //caffe_set(n_pred, Dtype(this->assign_counts_full_.mutable_cpu_data()), ac_full);
  caffe_set(n_pred, Dtype(0), losses);
  
  for (int i = 0; i < n_pred; ++i) {
	//ac_full[i] = this->assign_counts_full_.mutable_cpu_data()[i];
  }
	
  for (int i = 0; i < num; ++i) {
	Dtype loss_pred = 0;
	vector< pair<Dtype, int> > scores;

	int label = static_cast<int>(bottom_label[i]);
	for (int j = 0; j< n_pred; ++j) {
		const Dtype* bottom_data = bottom[j]->cpu_data();
		Dtype prob = std::max(
			bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		loss_pred = -log(prob);
		scores.push_back(make_pair(loss_pred, j));
	}

	std::partial_sort(
	    scores.begin(), scores.begin() + k,
	    scores.end(), std::less<std::pair<Dtype, int> >());
		
	for(int l = 0; l < k; l++){
		losses[scores[l].second] += scores[l].first;
		counts[scores[l].second]++;
		best[i*k+l] = scores[l].second;
	}
  }
		

  for(int i = 0; i < n_pred; i++){
	if(counts[i] > 0) 
		losses[i] /= counts[i];
		if(top.size() >= 2) {
			top[1]->mutable_cpu_data()[i] = counts[i];
		}
	//ac_full[i] += counts[i];
  	//top[2]->mutable_cpu_data()[i] = ac_full[i];
  	//this->assign_counts_full_.mutable_cpu_data()[i] += ac_full[i];
  	this->assign_counts_full_.mutable_cpu_data()[i] += counts[i];
  	top[2]->mutable_cpu_data()[i] = this->assign_counts_full_.mutable_cpu_data()[i];
  }
}

template <typename Dtype>
void ODMultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int n_pred = bottom.size()-1;
  int k =  this->layer_param_.mcl_param().hard_k();
  
  Dtype* best = this->best_pred_.mutable_cpu_data();
	
  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  const	Dtype* counts = this->assign_counts_.cpu_data();
  const Dtype* losses = top[0]->cpu_diff();

  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
	

  //For each predictor in the ensemble
  for (int j=0; j<n_pred; ++j){
	//Get predictions from predictor j
    const Dtype* bottom_data = bottom[j]->cpu_data();
			
	//Clear diff blob
    Dtype* bottom_diff = bottom[j]->mutable_cpu_diff();
    caffe_set(bottom[j]->count(), Dtype(0), bottom_diff);
      
	//Compute loss scale (adjusted for network parameters)
	Dtype scale = counts[j] > 0 ? -losses[j]/counts[j] : 0;
			
	//Pass back gradient at gradient scale / predicted prob
    for (int i = 0; i < num; ++i){				
		for(int l = 0; l < k; l++) 
			if(best[i*k+l] == j){
	 			int label = static_cast<int>(bottom_label[i]);
	   			Dtype prob = std::max(
						bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
				    	bottom_diff[i * dim + label] =  scale / prob;
			}
	}
  }
}

INSTANTIATE_CLASS(ODMultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(ODMultinomialLogisticLoss);
}  // namespace caffe
