#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include <iostream>
//#include <chrono>
#include <time.h>
//#include <random>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void ODStochasticPoolingMultinomialLogisticLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	this->assign_counts_full_.Reshape(bottom[0]->num(), 1, 1, 1);
  	//float start_rand_award = this->layer_param_.od_param().start_rand_award; // $b \in \{.1,.2,.3,.4.,.5\}$
  	this->update_k_models_.Reshape(1,1,1,1);
  	this->update_k_models_.mutable_cpu_data()[0] = this->layer_param_.odsp_param().update_k_models();
  	this->temp_.Reshape(1,1,1,1);
  	this->temp_.mutable_cpu_data()[0] = this->layer_param_.odsp_param().temp();
	
}


//Multinomial Loss for Random Award B Variation for 'Optimal Dieting'
template <typename Dtype>
void ODStochasticPoolingMultinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	int k = this->update_k_models_.mutable_cpu_data()[0]; 	

	//this->best_pred_.Reshape(bottom[0]->num(), this->layer_param_.mcl_param().hard_k(), 1, 1);
	this->best_pred_.Reshape(bottom[0]->num(), k, 1, 1);

	this->assign_counts_.Reshape(bottom.size()-1, 1, 1, 1);
	top[0]->Reshape(bottom.size()-1, 1, 1, 1);
	if(top.size() >= 2) {
		top[1]->Reshape(bottom.size()-1,1,1,1);
		top[2]->Reshape(bottom.size()-1,1,1,1);
	}
   	top[3]->Reshape(1,1,1,1);
	CHECK_EQ(bottom[bottom.size()-1]->channels(), 1);
	CHECK_EQ(bottom[bottom.size()-1]->height(), 1);
	CHECK_EQ(bottom[bottom.size()-1]->width(), 1);
}



template <typename Dtype>
void ODStochasticPoolingMultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int n_pred = bottom.size()-1;

  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int k = this->update_k_models_.mutable_cpu_data()[0];
  int temp = this->temp_.mutable_cpu_data()[0];
  float temp_decay = this->layer_param_.odsp_param().temp_decay();
  
  //************OD Random Award Parameters*********************
  //rand_select : if true, select model to update randomly. Else select 2nd best
  //decrease_rand : anneal b over time

  Dtype* best = this->best_pred_.mutable_cpu_data();
  Dtype* counts = this->assign_counts_.mutable_cpu_data();
  Dtype* losses = top[0]->mutable_cpu_data();
  
  caffe_set(n_pred, Dtype(0), counts);
  caffe_set(n_pred, Dtype(0), losses);
  
  for (int i = 0; i < num; ++i) {
	Dtype loss_pred = 0;
	vector< pair<Dtype, int> > scores;

	//for instance i, record all losses for all j labels
	int label = static_cast<int>(bottom_label[i]);
	//first calc denominator for softmax
	float denom = 0;
	for (int j = 0; j< n_pred; ++j) {
		const Dtype* bottom_data = bottom[j]->cpu_data();
		Dtype prob = std::max(
                        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		denom += exp(prob / temp);
	}
	for (int j = 0; j< n_pred; ++j) {
		const Dtype* bottom_data = bottom[j]->cpu_data();
		//Dtype prob = bottom_data[i * dim + label];
		Dtype prob = std::max(
                        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		float scaled_prob = exp(prob/temp) / denom;
		scores.push_back(make_pair(scaled_prob, j));
		
	}
	vector<int> added;
	while(added.size() < k) {
		float p_sum = 0.0; //helper var for determining which model to update
		float p = ((float) rand() / (RAND_MAX)); 
		for (int l=0; l < scores.size(); l++) {
			p_sum += scores[l].first;
			if (p < p_sum && std::find(added.begin(), added.end(), scores[l].second) == added.end()) {
				added.push_back(scores[l].second);
				break;
			}
		}
		
	}
	for(int l = 0; l < k; l++){
 		//compute log loss
		const Dtype* bottom_data = bottom[added[l]]->cpu_data();
		Dtype prob = std::max(
                        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		loss_pred = -log(prob);
		losses[added[l]] += loss_pred;
		counts[added[l]]++;
		best[i*k+l] = added[l];

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
  this->temp_.mutable_cpu_data()[0] *= (1-temp_decay);
  top[3]->mutable_cpu_data()[0] = k;
  //TODO : anneal k every...?
}

template <typename Dtype>
void ODStochasticPoolingMultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int n_pred = bottom.size()-1;
  //ADDED
  int k = this->update_k_models_.mutable_cpu_data()[0];
  
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

INSTANTIATE_CLASS(ODStochasticPoolingMultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(ODStochasticPoolingMultinomialLogisticLoss);
}  // namespace caffe
