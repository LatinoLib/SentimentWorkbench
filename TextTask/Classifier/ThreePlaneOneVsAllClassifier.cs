using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;

namespace TextTask.Classifier
{
    [Obsolete]
    public sealed class ThreePlaneOneVsAllClassifier : BaseClassifier<SentimentLabel>
    {
        private Model mPosModel;
        private Model mNegModel;
        private Model mNeuModel;

        public ThreePlaneOneVsAllClassifier()
        {
            NumTrainFolds = 2;
        }

        public ThreePlaneOneVsAllClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public int NumTrainFolds { get; set; }

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            Preconditions.CheckNotNull(dataset);

            var ds = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset
                .Select(le => new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label, le.Example)));

            mPosModel = TrainModel(ds, SentimentLabel.Positive, SentimentLabel.Negative, SentimentLabel.Neutral);
            mNegModel = TrainModel(ds, SentimentLabel.Negative, SentimentLabel.Positive, SentimentLabel.Neutral);
            mNeuModel = TrainModel(ds, SentimentLabel.Neutral, SentimentLabel.Positive, SentimentLabel.Negative);

            IsTrained = true;
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(IsTrained);

            var modelPredictions = new[] 
                {
                    new ModelPrediction(mPosModel, example), 
                    new ModelPrediction(mNegModel, example), 
                    new ModelPrediction(mNeuModel, example)
                };

            IGrouping<SentimentLabel, ModelPrediction.Score> first = modelPredictions
                .SelectMany(mp => mp.Scores)
                .GroupBy(s => s.Label)
                .OrderByDescending(g => g.Sum(s => s.VoteValue))
                .ThenByDescending(g => g.Sum(s => s.Owner.Prediction.BestScore/*s.Percentile*/))
                .First();

            return new Prediction<SentimentLabel>(new[] { new KeyDat<double, SentimentLabel>(first.Sum(s => s.VoteValue), first.Key) });
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[]
                {
                    mNegModel != null ? mNegModel.InnerModel : null, 
                    mNeuModel != null ? mNeuModel.InnerModel : null, 
                    mPosModel != null ? mPosModel.InnerModel : null
                }.Where(c => c is IDisposable).Cast<IDisposable>();
        }

        private Model TrainModel(LabeledDataset<SentimentLabel, SparseVector<double>> dataset, 
            SentimentLabel label, SentimentLabel otherLabel1, SentimentLabel otherLabel2)
        {
            IModel<SentimentLabel, SparseVector<double>> model = CreateModel();

            var otherLabelWeight1 = (double)dataset.Count(le => le.Label == otherLabel1) / dataset.Count(le => le.Label != label);
            var otherLabelWeight2 = (double)dataset.Count(le => le.Label == otherLabel2) / dataset.Count(le => le.Label != label);

            dataset = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Select(le =>
                new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label == label ? label : otherLabel1, le.Example)));

            var scores = new List<double>();
            var scoresOthers = new List<double>();
            var validation = new CrossValidator<SentimentLabel, SparseVector<double>>
                {
                    NumFolds = NumTrainFolds,
                    Dataset = dataset,

                    OnAfterPrediction = (sender, foldN, m, ex, le, prediction) =>
                    {
                        if (le.Label == prediction.BestClassLabel)
                        {
                            if (prediction.BestClassLabel == label)
                            {
                                scores.Add(prediction.BestScore);
                            }
                            else 
                            {
                                scoresOthers.Add(prediction.BestScore);
                            }
                        }
                        return true;
                    }
                };
            validation.Models.Add(model);
            validation.Run();

            // train model
            model.Train(dataset);

            return new Model
                {
                    InnerModel = model,
                    Weight = validation.PerfData.GetSumPerfMatrix(validation.ExpName, validation.GetModelName(model)).GetMacroF1(),
                    Label = label,
                    OtherLabel1 = otherLabel1,
                    OtherLabelWeight1 = otherLabelWeight1,
                    OtherLabel2 = otherLabel2,
                    OtherLabelWeight2 = otherLabelWeight2,
                    Scores = scores.OrderBy(s => s).ToArray(),
                    ScoresOthers = scoresOthers.OrderBy(s => s).ToArray()
                };
        }

        private static double GetPercentile(double score, double[] scores)
        {
            return (double)Math.Abs(Array.BinarySearch(scores, score)) / scores.Length;
        }

        /*
        public override void Save(BinarySerializer writer)
        {
            mPosModel.Save(writer);
            mNeuModel.Save(writer);
            mNegModel.Save(writer);
            writer.WriteBool(IsTrained);
        }

        public override sealed void Load(BinarySerializer reader)
        {
            mPosModel = new SvmBinaryClassifier<SentimentLabel>(reader);
            mNeuModel = new SvmBinaryClassifier<SentimentLabel>(reader);
            mNegModel = new SvmBinaryClassifier<SentimentLabel>(reader);
            IsTrained = reader.ReadBool();
        }
*/

        private class Model
        {
            public IModel<SentimentLabel, SparseVector<double>> InnerModel { get; set; }
            public double Weight { get; set; }
            public SentimentLabel Label { get; set; }
            public SentimentLabel OtherLabel1 { get; set; }
            public SentimentLabel OtherLabel2 { get; set; }
            public double OtherLabelWeight1 { get; set; }
            public double OtherLabelWeight2 { get; set; }

            public double[] Scores { get; set; }
            public double[] ScoresOthers { get; set; }
        }

        private class ModelPrediction
        {
            public ModelPrediction(Model model, SparseVector<double> example)
            {
                Model = model;
                Prediction = model.InnerModel.Predict(example);
            }

            public Model Model { get; set; }
            public Prediction<SentimentLabel> Prediction { get; set; }
            public Score[] Scores
            {
                get
                {
                    return Prediction.BestClassLabel == Model.Label
                        ? new[] { new Score { Owner = this, Label = Model.Label, VoteValue = Model.Weight * 1 } }
                        : new[]
                            {
                                new Score { Owner = this, Label = Model.OtherLabel1, VoteValue = Model.OtherLabelWeight1 }, 
                                new Score { Owner = this, Label = Model.OtherLabel2, VoteValue = Model.OtherLabelWeight2 }
                            };
                }
            }

            public class Score
            {
                public ModelPrediction Owner { get; set; }
                public SentimentLabel Label { get; set; }
                public double VoteValue { get; set; }
                public double Percentile
                {
                    get
                    {
                        return Label == Owner.Model.Label 
                            ? GetPercentile(Owner.Prediction.BestScore, Owner.Model.Scores)
                            : GetPercentile(Owner.Prediction.BestScore, Owner.Model.ScoresOthers);
                    }
                }
            }
        }
    }
}