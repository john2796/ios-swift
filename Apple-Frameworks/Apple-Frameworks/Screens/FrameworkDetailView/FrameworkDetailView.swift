//
//  FrameworkDetailView.swift
//  Apple-Frameworks
//
//  Created by john benedict miranda on 1/5/24.
//

import SwiftUI

struct FrameworkDetailView: View {
    var framework: Framework
    @State private var isShowingSafariView = false
    
    var body: some View {
        VStack {
            FrameworkTitleView(framework: framework)
            
            Text(framework.description)
                .font(.body)
                .padding()
            
            Spacer()
            
            Button {
                isShowingSafariView = true
            } label: {
                Label("Learn More", systemImage: "book.fill")
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .tint(.red)
            }
            .sheet(isPresented: $isShowingSafariView, content: {
                SafariView(url: URL(string: framework.urlString) ?? URL (string: "www.apple.com")!)
            })
        }
    }
}
