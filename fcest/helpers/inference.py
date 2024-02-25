import logging

import gpflow
from gpflow.monitor import (
    # ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
import tensorflow as tf


def run_adam(
    model_type: str,
    model,
    iterations: int,
    log_interval: int,
    log_dir: str = None,
    data: tuple = None,
    train_inducing_variables: bool = True,
) -> list:
    """
    GPflow utility function for running the Adam optimizer.
    View Tensorboard logs by running `tensorboard --logdir=${log_dir}`.

    TODO: allow for minibatch training

    Notes:
    - The Adam optimizer is typically used in deep neural networks.
        It may not be the ideal optimizer for GP models.

    Parameters
    ----------
    :param model_type:
    :param model:
        GPflow model.
    :param iterations:
        The number of iterations.
    :param log_interval:
    :param log_dir:
    :param data:
        A tuple with:
            x_observed: expected in shape (n_time_steps, 1), i.e. (N, 1).
            y_observed: expected in shape (n_time_steps, n_time_series), i.e. (N, D).
    :param train_inducing_variables:
        Whether to train inducing variables.
    :return:
    """
    if not train_inducing_variables:
        gpflow.set_trainable(model.inducing_variable, False)

    # Create an Adam Optimizer action
    logf = []
    match model_type:
        case "SVWP":
            training_loss = model.training_loss_closure(
                data,
                compile=True  # compile=True (default): compiles using tf.function
            )
        case "VWP":
            # train_iter = iter(train_dataset.batch(minibatch_size))
            training_loss = model.training_loss_closure(
                # train_iter,
                compile=True,
            )

    optimizer = tf.optimizers.Adam(
        learning_rate=0.001
    )

    # Set up Tensorboard monitoring.
    if log_dir is not None:

        model_task = ModelToTensorBoard(log_dir, model)
        # image_task = ImageToTensorBoard(log_dir, plot_prediction, "image_samples")

        match model_type:
            case "SVWP":
                lml_task = ScalarToTensorBoard(
                    log_dir,
                    lambda: model.training_loss(data=data),
                    "training_objective"
                )
            case "VWP":
                lml_task = ScalarToTensorBoard(
                    log_dir,
                    lambda: model.training_loss(),
                    "training_objective"
                )

        # Plotting tasks can be quite slow. We want to run them less frequently.
        # We group them in a `MonitorTaskGroup` and set the period to 5.
        # slow_tasks = MonitorTaskGroup(image_task, period=5)

        # The other tasks are fast. We run them at each iteration of the optimization.
        fast_tasks = MonitorTaskGroup(
            [model_task, lml_task],
            period=1
        )

        # Both groups are passed to the monitor.
        # `slow_tasks` will be run five times less frequently than `fast_tasks`.
        # monitor = Monitor(fast_tasks, slow_tasks)
        monitor = Monitor(fast_tasks)

    @tf.function
    def optimization_step():
        optimizer.minimize(
            training_loss,
            model.trainable_variables
        )

    for step in range(iterations):
        optimization_step()
        if log_dir is not None:
            monitor(step)  # run monitoring to Tensorboard
        if step % log_interval == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            logging.info(f"Epoch {step:04d}: ELBO (train) {elbo:.2f}")
    return logf
