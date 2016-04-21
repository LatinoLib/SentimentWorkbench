using System;
using System.Runtime.CompilerServices;

namespace TextTask
{
    /// <summary>
    /// source http://www.codeproject.com/Tips/293072/Using-reference-counting-on-IDisposable-objects
    /// </summary>

    public static class DisposableExtensions
    {
        /// <summary>
        /// Values in a ConditionalWeakTable need to be a reference type,
        /// so box the refcount int in a class.
        /// </summary>
        private class RefCount
        {
            public int Count;
        }

        private static readonly ConditionalWeakTable<IDisposable, RefCount> mRefCounts =
            new ConditionalWeakTable<IDisposable, RefCount>();

        /// <summary>
        /// Extension method for IDisposable.
        /// Increments the Count for the given IDisposable object.
        /// Note: newly instantiated objects don't automatically have a Count of 1!
        /// If you wish to use ref-counting, always call retain() whenever you want
        /// to take ownership of an object.
        /// </summary>
        /// <remarks>This method is thread-safe.</remarks>
        /// <param name="disposable">The disposable that should be retained.</param>
        public static void Retain(this IDisposable disposable)
        {
            lock (mRefCounts)
            {
                RefCount refCount = mRefCounts.GetOrCreateValue(disposable);
                refCount.Count++;
            }
        }

        /// <summary>
        /// Extension method for IDisposable.
        /// Decrements the Count for the given disposable.
        /// </summary>
        /// <remarks>This method is thread-safe.</remarks>
        /// <param name="disposable">The disposable to release.</param>
        public static void Release(this IDisposable disposable)
        {
            lock (mRefCounts)
            {
                RefCount refCount = mRefCounts.GetOrCreateValue(disposable);
                if (refCount.Count > 0)
                {
                    refCount.Count--;
                    if (refCount.Count == 0)
                    {
                        mRefCounts.Remove(disposable);
                        disposable.Dispose();
                    }
                }
                else
                {
                    // Retain() was never called, so assume there is only
                    // one reference, which is now calling Release()
                    disposable.Dispose();
                }
            }
        }
    }
}