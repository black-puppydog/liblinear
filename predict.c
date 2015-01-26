#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"

#ifndef _DENSE_REP
struct feature_node *x;
#endif
int max_nr_attr = 64;

struct model* model_;
int flag_predict_probability=0;

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void do_predict(FILE *input, FILE *output, struct model* model_)
{
	int correct = 0;
	int total;

	int nr_class=get_nr_class(model_);
	double *prob_estimates=NULL;
	double *x;
	int *y;
	int j, n, i, nr_f_check;
	int nr_feature=get_nr_feature(model_);
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	if(flag_predict_probability)
	{
		int *labels;

		if(model_->param.solver_type!=L2R_LR)
		{
			fprintf(stderr, "probability output is only supported for logistic regression\n");
			exit(1);
		}

		labels=(int *) malloc(nr_class*sizeof(int));
		get_labels(model_,labels);
		prob_estimates = (double *) malloc(nr_class*sizeof(double));
		fprintf(output,"labels");		
		for(j=0;j<nr_class;j++)
			fprintf(output," %d",labels[j]);
		fprintf(output,"\n");
		free(labels);
	}
	
	fread((void*)(&total), sizeof(total), 1, input);
	fread((void*)(&nr_f_check), sizeof(nr_f_check), 1, input);
	if(nr_f_check!=n)
	  exit_input_error(2);
	x = (double*)malloc(n*sizeof(double));
	
	y = (int*)malloc(total*sizeof(int));
	fread((void*)(y), sizeof(y[0]), total, input);

	for(i=0; i<total; i++)
	{
		int target_label, predict_label;

		target_label = y[i];	
		fread((void*)x, sizeof(double), nr_feature, input);
		
		
		if(model_->bias>=0)
			x[n-1] = model_->bias;

		if(flag_predict_probability)
		{
			int j;
			predict_label = predict_probability(model_,x,prob_estimates);
			fprintf(output,"%d",predict_label);
			for(j=0;j<model_->nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = predict(model_,x);
			fprintf(output,"%d\n",predict_label);
		}

		if(predict_label == target_label)
			++correct;
	}
	//printf("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
	if(flag_predict_probability)
		free(prob_estimates);
}

void exit_with_help()
{
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				flag_predict_probability = atoi(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	if(i>=argc)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model_=load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

#ifndef _DENSE_REP
	x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
#endif
	do_predict(input, output, model_);
	destroy_model(model_);
	free(line);
#ifndef _DENSE_REP
	free(x);
#endif
	fclose(input);
	fclose(output);
	return 0;
}

