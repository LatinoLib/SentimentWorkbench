using System;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.TextMining;
using TextTask.DataSource;

namespace TextTask
{
    public class TaskContext : IDisposable
    {
        public LabeledTextSource DataSource { get; set; }
        public TextFeatureProcessor FeatureProcessor { get; set; }
        public LabeledDataset<SentimentLabel, SparseVector<double>> LabeledBowDataset { get; set; }

        public BowSpace BowSpace { get; set; }
        public Func<BowSpace> BowSpaceFactory { get; set; }

        public IModel<SentimentLabel, SparseVector<double>>[] Models { get; set; }
        public delegate IModel<SentimentLabel, SparseVector<double>> ModelFactoryFunc(int modelIdx);
        public ModelFactoryFunc ModelFactory { get; set; }

        public delegate string ModelNameHandler(IModel<SentimentLabel, SparseVector<double>> model);
        public ModelNameHandler ModelNameFunc { get; set; }

        public virtual string GetModelName(IModel<SentimentLabel, SparseVector<double>> model)
        {
            Preconditions.CheckNotNull(model);
            return ModelNameFunc != null ? ModelNameFunc(model) : model.GetType().Name;
        }

        public virtual string GetModelName(int modelIdx)
        {
            Preconditions.CheckNotNull(Models);
            Preconditions.CheckArgumentRange(modelIdx >= 0 && modelIdx < Models.Length);
            Preconditions.CheckArgument(Models[modelIdx] != null || ModelFactory != null);
            return GetModelName(Models[modelIdx] ?? ModelFactory(modelIdx));
        }

        public void Dispose()
        {
            if (DataSource != null)
            {
                DataSource.Dispose();
            }
            if (Models != null)
            {
                foreach (IDisposable disposable in Models.OfType<IDisposable>())
                {
                    disposable.Dispose();
                }
            }
        }
    }
}