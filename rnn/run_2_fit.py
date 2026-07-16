import os
import random

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from rnn.utils.const import *


def main():
    try:
        # determinism
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        g = torch.Generator()
        g.manual_seed(SEED)

        def seed_worker(worker_id):
            worker_seed = SEED + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        # load datasets
        train_dataset = torch.load(f"{DATASET_DIR}train{DATASET_POSTFIX}", weights_only=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker,
                                  generator=g)
        val_dataset = torch.load(f"{DATASET_DIR}val{DATASET_POSTFIX}", weights_only=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(val_dataset) > 0 else None
        # test_dataset = torch.load(f"{DATASET_DIR}test{DATASET_POSTFIX}", weights_only=False)
        # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print("Starting training loop...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = MODEL(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL,
            n_heads=N_HEADS).to(device)

        # Toto aktivuje Triton backend (default='inductor') pro rychlejší běh
        print("Compiling model...")
        model = torch.compile(model)

        criterion = CRITERION
        optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=SCHEDULER_PATIENCE,
            factor=SCHEDULER_FACTOR,
            threshold=SCHEDULER_THRESHOLD,
            min_lr=SCHEDULER_MIN_LR)

        train_loss_history = []
        val_loss_history = []

        best_val_loss = np.inf  # our metric to save the best model
        for epoch in range(EPOCHS):
            # switch to train mode
            model.train()
            total_train_loss = 0

            for sequences, labels in train_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)  # Store loss

            # Validation phase
            if val_loader:
                # switch to val mode (turn off dropout etc.)
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for sequences, labels in val_loader:
                        sequences = sequences.to(device)
                        labels = labels.to(device)

                        outputs = model(sequences)
                        loss = criterion(outputs, labels)
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                val_loss_history.append(avg_val_loss)  # Store loss

                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr != old_lr:
                    print(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")

                print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

                # Save the best model
                if avg_val_loss < best_val_loss:
                    best_model_filename = f"{MODEL_SAVE_PATH}.best"
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), best_model_filename)
                    print(f"New best model (Val Loss: {avg_val_loss:.6f}) to '{best_model_filename}'")
            else:
                # In case there's no validation set...
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}')

        print("\tTRAIN LOOP FINISHED")
        if not val_loader:
            last_epoch_filename = f"{MODEL_SAVE_PATH}.last_epoch"
            torch.save(model.state_dict(), last_epoch_filename)
            print(f"Model (last epoch) saved do '{last_epoch_filename}")

        print("Generating training loss plot...")
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss')
        if val_loader:
            plt.plot(val_loss_history, label='Validation Loss')
        plt.title('Training & Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        # log y scale for better visibility
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Ensure 'temp' directory exists
        os.makedirs('temp', exist_ok=True)
        plot_filename = "temp/training_loss_plot.png"
        plt.savefig(plot_filename)
        print(f"Training plot saved to: {plot_filename}")

    except Exception as e:
        print(f"\nA serious error occurred: {e}")


if __name__ == "__main__":
    main()
