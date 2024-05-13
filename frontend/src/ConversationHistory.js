import React from 'react';

function ConversationHistory({ history }) {
    return (
        <div className="conversation-history">
            {history.map((entry, index) => (
                <div key={index} className="conversation-entry">
                    <div className="user-prompt">{entry.userPrompt}</div>
                    <div className="assistant-response">{entry.assistantResponse}</div>
                </div>
            ))}
        </div>
    );
}

export default ConversationHistory;