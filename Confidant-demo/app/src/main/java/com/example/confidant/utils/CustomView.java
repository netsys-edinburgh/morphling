package com.example.confidant.utils;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.confidant.R;

import java.util.List;

public class CustomView {
    public static class LogItem {
        private String message;

        public LogItem(String message) {
            this.message = message;
        }

        public String getMessage() {
            return message;
        }
    }

    public static class LogAdapter extends RecyclerView.Adapter<LogAdapter.ViewHolder> {
        private List<LogItem> logs;

        public LogAdapter(List<LogItem> logs) {
            this.logs = logs;
        }

        public void addLog(LogItem log, RecyclerView logView) {
            logs.add(log);
            notifyItemInserted(logs.size() - 1);
            logView.scrollToPosition(logs.size() - 1);
        }
        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.log_item, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            LogItem log = logs.get(position);
            holder.message.setText(log.getMessage());
        }

        @Override
        public int getItemCount() {
            return logs.size();
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            public TextView message;

            public ViewHolder(@NonNull View itemView) {
                super(itemView);
                message = itemView.findViewById(R.id.message);
            }
        }
    }
}
