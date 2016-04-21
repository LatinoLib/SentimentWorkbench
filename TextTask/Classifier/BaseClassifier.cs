using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public abstract class BaseClassifier<LabelT> : IModel<LabelT, SparseVector<double>>, IDisposable
    {
        public Func<IModel<LabelT, SparseVector<double>>> InnerModelFunc { get; set; }
        public Type RequiredExampleType { get { return null; } }
        public bool IsTrained { get; protected set; }

        public virtual void Save(BinarySerializer writer)
        {
            throw new NotImplementedException();
        }

        public virtual void Load(BinarySerializer reader)
        {
            throw new NotImplementedException();
        }

        public abstract void Train(ILabeledExampleCollection<LabelT, SparseVector<double>> dataset);

        public abstract Prediction<LabelT> Predict(SparseVector<double> example);

        public void Train(ILabeledExampleCollection<LabelT> dataset)
        {
            Train((ILabeledExampleCollection<LabelT, SparseVector<double>>)dataset);
        }

        public Prediction<LabelT> Predict(object example)        
        {
            return Predict((SparseVector<double>)example);
        }

        protected IModel<LabelT, SparseVector<double>> CreateModel()
        {
            Preconditions.CheckNotNull(InnerModelFunc);
            return InnerModelFunc();
        }

        protected static readonly IEnumerable<IDisposable> EmptyDisposables = Enumerable.Empty<IDisposable>();

        protected abstract IEnumerable<IDisposable> GetDisposables();

        public void Dispose()
        {
            foreach (IDisposable disposable in GetDisposables())
            {
                disposable.Dispose();
            }
        }
    }
}