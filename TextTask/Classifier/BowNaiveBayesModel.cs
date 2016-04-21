using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class BowNaiveBayesModel<LblT> : BaseClassifier<LblT>
    {
        private readonly NaiveBayesClassifier<LblT> mModel;

        public BowNaiveBayesModel()
        {
            mModel = new NaiveBayesClassifier<LblT>();
        }

        public BowNaiveBayesModel(NaiveBayesClassifier<LblT> model)
        {
            mModel = Preconditions.CheckNotNull(model);
        }

        public override void Train(ILabeledExampleCollection<LblT, SparseVector<double>> dataset)
        {
            var ds = (LabeledDataset<LblT, SparseVector<double>>)dataset;
            mModel.Train((LabeledDataset<LblT, BinaryVector>)ds.ConvertDataset(typeof(BinaryVector), false));
        }

        public override Prediction<LblT> Predict(SparseVector<double> example)
        {
            return mModel.Predict((BinaryVector)ModelUtils.ConvertExample(example, typeof(BinaryVector)));
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return EmptyDisposables;
        }
    }
}