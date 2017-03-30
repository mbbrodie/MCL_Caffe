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
void MCLDecForDivLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	this->assign_counts_full_.Reshape(bottom[0]->num(), 1, 1, 1);
  	//float start_rand_award = this->layer_param_.od_param().start_rand_award; // $b \in \{.1,.2,.3,.4.,.5\}$
  	this->rand_award_prob_.Reshape(1,1,1,1);
  	this->rand_award_prob_.mutable_cpu_data()[0] = this->layer_param_.od_param().start_rand_award();
}


//Multinomial Loss for Random Award B Variation for 'Optimal Dieting'
template <typename Dtype>
void MCLDecForDivLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	  int n_pred = bottom.size()-1;
	  int k =  this->layer_param_.mcl_param().hard_k();
	  bool draw_m_minus_one = this->layer_param_.od_param().draw_m_minus_one();
	  bool award_all = this->layer_param_.od_param().award_all();
	  if (award_all || draw_m_minus_one) {
		k = n_pred;	
	  }
	  else {
		k = 2;
	  }

	//this->best_pred_.Reshape(bottom[0]->num(), this->layer_param_.mcl_param().hard_k(), 1, 1);
  int n_outputs = 10; ///FIGURE OUT A WAY TO GET THIS INFO DYNAMICALLY!!!!
	//this->best_pred_.Reshape(bottom[0]->num(), k, 1, 1);
  this->best_pred_.Reshape(bottom[0]->num(), n_outputs, 1, 1);

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
void MCLDecForDivLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int n_pred = bottom.size()-1;
  int n_outputs = 10; ///FIGURE OUT A WAY TO GET THIS INFO DYNAMICALLY!!!!

  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int k =  this->layer_param_.mcl_param().hard_k();

  //************OD Random Award Parameters*********************
  //rand_select : if true, select model to update randomly. Else select 2nd best
  //decrease_rand : anneal b over time
  float b = this->rand_award_prob_.mutable_cpu_data()[0];
  bool rand_select = this->layer_param_.od_param().rand_select();
  bool decrease_rand = this->layer_param_.od_param().decrease_rand();
  bool draw_m_minus_one = this->layer_param_.od_param().draw_m_minus_one();
  float rand_decay = this->layer_param_.od_param().rand_decay();  
  bool award_all = this->layer_param_.od_param().award_all();
  if (award_all || draw_m_minus_one) {
	k = n_pred;	
  }
  else {
	k = 2;
  }


  Dtype* best = this->best_pred_.mutable_cpu_data();
 std::cout << best[0] << "\n";
  Dtype* counts = this->assign_counts_.mutable_cpu_data();
  Dtype* losses = top[0]->mutable_cpu_data();
  
  caffe_set(n_pred, Dtype(0), counts);
  caffe_set(n_pred, Dtype(0), losses);
  
  for (int i = 0; i < num; ++i) {
	Dtype loss_pred = 0;
	vector< pair<Dtype, int> > scores;

	//for instance i, record all losses for all j labels
	int label = static_cast<int>(bottom_label[i]);
	for (int j = 0; j< n_pred; ++j) {
		const Dtype* bottom_data = bottom[j]->cpu_data();
		int max_label = -1;
                Dtype max_prob = -1;
                for (int l=0; l< n_outputs; l++) {
			if (bottom_data[i * dim + l] > max_prob) {
				max_label = l;
				max_prob = bottom_data[i * dim + l];
			}	
		}
		if (label == max_label || j == n_pred-1) {
			Dtype prob = std::max(
                        	bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
			loss_pred = -log(prob);
			losses[j] += loss_pred;
			counts[j]++;
			best[i*k+label] = j;
			break;
			//scores.push_back(make_pair(loss_pred, j));
		}
	}

	//random seed info
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//unsigned int seed = static_cast<unsigned int>( time(NULL) );
	//std::default_random_engine generator (seed);
	//std::uniform_real_distribution<float> distribution (0.0,1.0);

/*
	float award_all_prob = ((float) rand() / (RAND_MAX)); //distribution(generator)

		//sort and keep the top score
	std::partial_sort(
	    scores.begin(), scores.begin() + k,
	    scores.end(), std::less<std::pair<Dtype, int> >());
		
	for(int l = 0; l < k; l++){
		if (l == 0) {
			losses[scores[l].second] += scores[l].first;
			counts[scores[l].second]++;
			best[i*k+l] = scores[l].second;
		}
		else {
			if (award_all) {
				if (draw_m_minus_one) {
					float p = ((float) rand() / (RAND_MAX)); //distribution(generator);
					if (p < b) {
						losses[scores[l].second] += scores[l].first;
						counts[scores[l].second]++;
						best[i*k+l] = scores[l].second;
					}	
					else {
						best[i*k+l] = -1; // -1 means DO NOT UPDATE in Backprop below
					}
				}
				else {
					if (award_all_prob < b) {
						losses[scores[l].second] += scores[l].first;
						counts[scores[l].second]++;
						best[i*k+l] = scores[l].second;
					}
					else {
						best[i*k+l] = -1; 
					}
				}
			}
			else {
				float p = ((float) rand() / (RAND_MAX)); //distribution(generator);
				if(!rand_select) {
					if (p < b) {
						losses[scores[l].second] += scores[l].first;
						counts[scores[l].second]++;
						best[i*k+l] = scores[l].second;
					}
					else {
						best[i*k+l] = -1;
					}
				}
				else {
					if (p < b) {	
						//std::uniform_int_distribution<> distr (1,n_pred-1);
						int m_to_update = ((int) rand() % (n_pred-1) + 1); //distr(generator);
						losses[scores[m_to_update].second] += scores[m_to_update].first;
						counts[scores[m_to_update].second]++;
						best[i*k+l] = scores[m_to_update].second;
					}
					else {
						best[i*k+l] = -1; 
					}
					
				}
			}
			
		}

	}
*/
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
  if (decrease_rand) {
	this->rand_award_prob_.mutable_cpu_data()[0] *= (1-rand_decay);
  }
  top[3]->mutable_cpu_data()[0] = b;
}

template <typename Dtype>
void MCLDecForDivLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int n_pred = bottom.size()-1;
  int k =  this->layer_param_.mcl_param().hard_k();
  //ADDED
  bool draw_m_minus_one = this->layer_param_.od_param().draw_m_minus_one();
  bool award_all = this->layer_param_.od_param().award_all();
  if (award_all || draw_m_minus_one) {
	k = n_pred;	
  }
  else {
	k = 2;
  }
  
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

INSTANTIATE_CLASS(MCLDecForDivLossLayer);
REGISTER_LAYER_CLASS(MCLDecForDivLoss);
}  // namespace caffe
