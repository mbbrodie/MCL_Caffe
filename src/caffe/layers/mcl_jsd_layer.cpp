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
void MCLJSDLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	this->best_pred_.Reshape(bottom[0]->num(), (bottom.size()-1)/2, 1, 1);
	this->assign_counts_.Reshape((bottom.size()-1)/2, 1, 1, 1);
	top[0]->Reshape((bottom.size()-1)/2, 1, 1, 1);
	top[1]->Reshape((bottom.size()-1)/2,1,1,1);

	CHECK_EQ(bottom[(bottom.size()-1)/2]->channels(), 1);
	CHECK_EQ(bottom[(bottom.size()-1)/2]->height(), 1);
	CHECK_EQ(bottom[(bottom.size()-1)/2]->width(), 1);
}

template <typename Dtype>
void MCLJSDLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int n_pred = (bottom.size()-1) / 2;

  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int k_to_update =  (bottom.size()-1) /2; 
  Dtype* best = this->best_pred_.mutable_cpu_data();
  Dtype* counts = this->assign_counts_.mutable_cpu_data();
  Dtype* losses = top[0]->mutable_cpu_data();
  Dtype* divs = top[1]->mutable_cpu_data();
  
  caffe_set(n_pred, Dtype(0), counts);
  caffe_set(n_pred, Dtype(0), losses);
  
  for (int i = 0; i < num; ++i) {
	Dtype loss_pred = 0;
	Dtype div_pred = 0;
	vector< pair<Dtype, int> > scores;
	vector< pair<Dtype, int> > div_scores;

/*NOTE FOR CHRIS: I'm not actually doing anything with the diversity scores generated from the forward step.
All math that's used to update weights appears in the Backwards step*/
	int label = static_cast<int>(bottom_label[i]);
	for (int j = 0; j< n_pred; ++j) {
		const Dtype* bottom_data = bottom[j]->cpu_data();
		Dtype prob = std::max(
			bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		loss_pred = -log(prob); 
        div_pred = 0;
        for (int k=0; k < n_pred; ++k) {
            if (j==k) continue;
		    const Dtype* bottom_data_k = bottom[k]->cpu_data();
            div_pred += log(bottom_data[i * dim + label] / (0.5*(bottom_data[i * dim + label] + bottom_data_k[i * dim + label])));
        }
        div_pred *= ((1-prob) / (2*(n_pred-1))); 
        
		scores.push_back(make_pair(loss_pred, j));
		div_scores.push_back(make_pair(div_pred, j));
	}

	for(int l = 0; l < k_to_update; l++){
		losses[scores[l].second] += scores[l].first;
		divs[div_scores[l].second] += div_scores[l].first;
		counts[scores[l].second]++;
		best[i*k_to_update+l] = scores[l].second;
	}
  }
		
  for(int i = 0; i < n_pred; i++){
	if(counts[i] > 0) 
		losses[i] /= counts[i];
        divs[i] /= counts[i];
  }
}

template <typename Dtype>
void MCLJSDLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int n_pred = (bottom.size()-1)/2;
  int k = n_pred; 
  Dtype* best = this->best_pred_.mutable_cpu_data();
	
  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  const	Dtype* counts = this->assign_counts_.cpu_data();
  const Dtype* losses = top[0]->cpu_diff();
  const Dtype* divs = top[1]->cpu_data();

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
                Dtype div_pred = 0;
                for (int k2=0; k2 < n_pred; ++k2) {
                    if (j==k2) continue;
                    const Dtype* bottom_data_k = bottom[k2]->cpu_data();
                    div_pred += log(bottom_data[i * dim + label] / (0.5*(bottom_data[i * dim + label] + bottom_data_k[i * dim + label])));
                }
                div_pred *= ((1-prob) / (2*(n_pred-1))); //CHECK HERE - do we want to use the NLL? Or just 1-prob? (i.e. error between 0 and 1
                bottom_diff[i * dim + label] =  scale / prob - div_pred;//divs[j];
			}
	}
  }
}

INSTANTIATE_CLASS(MCLJSDLayer);
REGISTER_LAYER_CLASS(MCLJSD);
}  // namespace caffe
