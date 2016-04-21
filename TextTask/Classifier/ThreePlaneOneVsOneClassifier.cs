using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;

namespace TextTask.Classifier
{
    [Obsolete]
    public class ThreePlaneOneVsOneClassifier : BaseClassifier<SentimentLabel>
    {
        private Model mPosNegModel;
        private Model mPosNeuModel;
        private Model mNegNeuModel;

        public ThreePlaneOneVsOneClassifier()
        {
            NumTrainFolds = 2;
        }

        public ThreePlaneOneVsOneClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public int NumTrainFolds { get; set; }

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            Preconditions.CheckNotNull(dataset);

            var ds = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Where(le => le.Label != SentimentLabel.Neutral)
                .Select(le => new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label, le.Example)));
            mPosNegModel = TrainModel(ds, SentimentLabel.Positive, SentimentLabel.Negative);

            ds = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Where(le => le.Label != SentimentLabel.Positive)
                .Select(le => new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label, le.Example)));
            mNegNeuModel = TrainModel(ds, SentimentLabel.Negative, SentimentLabel.Neutral);

            ds = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Where(le => le.Label != SentimentLabel.Negative)
                .Select(le => new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label, le.Example)));
            mPosNeuModel = TrainModel(ds, SentimentLabel.Positive, SentimentLabel.Neutral);

            IsTrained = true;
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(IsTrained);

            var modelPredictions = new[]
                {
                    new ModelPrediction(mPosNegModel, example), 
                    new ModelPrediction(mNegNeuModel, example), 
                    new ModelPrediction(mPosNeuModel, example)
                };

            IGrouping<SentimentLabel, ModelPrediction.Score> first = modelPredictions
                .Select(mp => mp.ScoreValue)
                .GroupBy(s => s.Label)
                .OrderByDescending(g => g.Sum(s => s.VoteValue))
                .ThenByDescending(g => g.Sum(s => /*s.Owner.Prediction.BestScore */s.Percentile))
                .First();

            return new Prediction<SentimentLabel>(new[] { new KeyDat<double, SentimentLabel>(first.Sum(s => s.VoteValue), first.Key) });
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[]
                {
                    mNegNeuModel != null ? mNegNeuModel.InnerModel : null, 
                    mPosNegModel != null ? mPosNegModel.InnerModel : null, 
                    mPosNeuModel != null ? mPosNeuModel.InnerModel : null
                }.Where(c => c is IDisposable).Cast<IDisposable>();
        }

        private Model TrainModel(LabeledDataset<SentimentLabel, SparseVector<double>> dataset,
            SentimentLabel label1, SentimentLabel label2)
        {
            IModel<SentimentLabel, SparseVector<double>> model = CreateModel();
            var scores1 = new List<double>();
            var scores2 = new List<double>();

            var validation = new CrossValidator<SentimentLabel, SparseVector<double>>
            {
                NumFolds = NumTrainFolds,
                Dataset = dataset,

                OnAfterPrediction = (sender, foldN, m, ex, le, prediction) =>
                {
                    if (le.Label == prediction.BestClassLabel)
                    {
                        if (prediction.BestClassLabel == label1)
                        {
                            scores1.Add(prediction.BestScore);
                        }
                        else if (prediction.BestClassLabel == label2)
                        {
                            scores2.Add(prediction.BestScore);
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
                    Label1 = label1,
                    Label2 = label2,
                    Scores1 = scores1.OrderBy(s => s).ToArray(),
                    Scores2 = scores2.OrderBy(s => s).ToArray(),
                    Weight = validation.PerfData.GetSumPerfMatrix(validation.ExpName, validation.GetModelName(model)).GetMacroF1()
                };
        }

        public override void Save(BinarySerializer writer)
        {
            writer.WriteInt(NumTrainFolds);
            mPosNegModel.Save(writer);
            mNegNeuModel.Save(writer);
            mPosNeuModel.Save(writer);
            writer.WriteBool(IsTrained);
        }

        public override sealed void Load(BinarySerializer reader)
        {
            NumTrainFolds = reader.ReadInt();
            mPosNegModel = new Model(reader);
            mNegNeuModel = new Model(reader);
            mPosNeuModel = new Model(reader);
            IsTrained = reader.ReadBool();
        }

        private static double GetPercentile(double score, double[] scores)
        {
            return (double)Math.Abs(Array.BinarySearch(scores, score)) / scores.Length;
        }

        private class Model : ISerializable
        {
            public Model()
            {
                LabelVotingScores = new Dictionary<SentimentLabel, int>
                    {
                        { SentimentLabel.Positive, 1 },
                        { SentimentLabel.Neutral, 1 },
                        { SentimentLabel.Negative, 1 }
                    };
            }

            public Model(BinarySerializer reader)
            {
                Load(reader);
            }

            public IModel<SentimentLabel, SparseVector<double>> InnerModel { get; set; }
            public double Weight { get; set; }
            public SentimentLabel Label1 { get; set; }
            public SentimentLabel Label2 { get; set; }
            public double[] Scores1 { get; set; }
            public double[] Scores2 { get; set; }
            public Dictionary<SentimentLabel, int> LabelVotingScores { get; private set; }

            public void Save(BinarySerializer writer)
            {
                writer.WriteObject(InnerModel);
                writer.WriteDouble(Weight);
                writer.WriteValue(Label1);
                writer.WriteValue(Label2);

                writer.WriteInt(Scores1.Length);
                foreach (double s in Scores1)
                {
                    writer.WriteDouble(s);
                }

                writer.WriteInt(Scores2.Length);
                foreach (double s in Scores2)
                {
                    writer.WriteDouble(s);
                }

                LabelVotingScores.SaveDictionary(writer);
            }

            public void Load(BinarySerializer reader)
            {
                InnerModel = reader.ReadObject<IModel<SentimentLabel, SparseVector<double>>>();
                Weight = reader.ReadDouble();
                Label1 = reader.ReadValue<SentimentLabel>();
                Label2 = reader.ReadValue<SentimentLabel>();

                Scores1 = new double[reader.ReadInt()];
                for (int i = 0; i < Scores1.Length; i++)
                {
                    Scores1[i] = reader.ReadDouble();
                }

                Scores2 = new double[reader.ReadInt()];
                for (int i = 0; i < Scores2.Length; i++)
                {
                    Scores2[i] = reader.ReadDouble();
                }

                LabelVotingScores = reader.LoadDictionary<SentimentLabel, int>();
            }
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
            public Score ScoreValue
            {
                get
                {
                    return new Score
                        {
                            Owner = this, 
                            Label = Prediction.BestClassLabel,
                            VoteValue = Model.LabelVotingScores[Prediction.BestClassLabel]
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
                        return Label == Owner.Model.Label1
                            ? GetPercentile(Owner.Prediction.BestScore, Owner.Model.Scores1)
                            : GetPercentile(Owner.Prediction.BestScore, Owner.Model.Scores2);
                    }
                }
            }
        }
    }
}