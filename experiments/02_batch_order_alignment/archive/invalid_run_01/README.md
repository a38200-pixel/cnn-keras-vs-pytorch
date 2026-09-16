# Experiment 02 Attempt 1 - Invalid Run

## Invalid reason

Keras 3의 `fit()`은 첫 training epoch 전에 data adapter를 reset하며 이 과정에서 `Sequence.on_epoch_end()`를 호출한다. 기존 구현은 `on_epoch_end()`에서 다음 permutation으로 이동했기 때문에 Keras epoch 1은 schedule index 1을 사용했고, PyTorch epoch 1은 schedule index 0을 사용했다.

따라서 실제 training batch order가 Framework 간 동일하지 않았으며, 이 실행은 Experiment 02 Batch Order Alignment의 OFAT 결과로 사용할 수 없다.

이 디렉터리의 CSV, history, figure, model checkpoint, comparison summary와 batch-order artifact는 삭제하지 않고 technical debugging record로만 보존한다. 성능 수치와 Gap 변화는 최종 OFAT 표나 가설 평가에 사용하지 않는다.

수정된 active 코드는 order 변경을 `on_epoch_end()`에 의존하지 않고 실제 `on_epoch_begin()`에서 적용하며, 각 Framework/Seed/Epoch의 runtime order hash를 별도로 기록한다.
