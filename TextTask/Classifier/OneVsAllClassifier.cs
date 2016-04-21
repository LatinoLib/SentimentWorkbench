using System;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class OneVsAllClassifier<LblT, ExT> : IModel<LblT, ExT> 
    {
        private readonly IModel<LblT, ExT> mBinaryModel;

        public OneVsAllClassifier(LblT oneLabel, LblT otherLabel, IModel<LblT, ExT> binaryModel) 
        {
            Preconditions.CheckArgument(!oneLabel.Equals(otherLabel));
            mBinaryModel = Preconditions.CheckNotNull(binaryModel);
            OneLabel = oneLabel;
            OtherLabel = otherLabel;
        }

        public OneVsAllClassifier(LblT oneLabel, IModel<LblT, ExT> binaryModel) 
        {
            Preconditions.CheckArgument(typeof(LblT).IsEnum);
            Preconditions.CheckArgument(Enum.GetValues(typeof(LblT)).Length > 1);
            mBinaryModel = Preconditions.CheckNotNull(binaryModel);
            OneLabel = oneLabel;
            OtherLabel = Enum.GetValues(typeof(LblT)).Cast<LblT>().First(l => !l.Equals(oneLabel));
        }

        public Type RequiredExampleType { get { return typeof(ExT); } }
        public bool IsTrained { get; private set; }
        public LblT OneLabel { get; private set; }
        public LblT OtherLabel { get; private set; }

        public void Save(BinarySerializer writer)
        {
            throw new NotImplementedException();
        }

        public void Train(ILabeledExampleCollection<LblT> dataset)
        {
            Train((ILabeledExampleCollection<LblT, ExT>)dataset);
        }

        public Prediction<LblT> Predict(object example)
        {
            return Predict((ExT)example);
        }

        public void Train(ILabeledExampleCollection<LblT, ExT> dataset)
        {
            var binaryDataset = new LabeledDataset<LblT, ExT>(dataset.Select(le =>
                new LabeledExample<LblT, ExT>(le.Label.Equals(OneLabel) ? OneLabel : OtherLabel, le.Example)));

            mBinaryModel.Train(binaryDataset);
            IsTrained = true;
        }

        public Prediction<LblT> Predict(ExT example)
        {
            Preconditions.CheckState(IsTrained);
            return mBinaryModel.Predict(example);
        }
    }
}